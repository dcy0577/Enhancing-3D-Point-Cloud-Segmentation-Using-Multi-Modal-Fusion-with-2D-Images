# Code base on
# @article{thomas2019KPConv,
#     Author = {Thomas, Hugues and Qi, Charles R. and Deschaud, Jean-Emmanuel and Marcotegui, Beatriz and Goulette, Fran{\c{c}}ois and Guibas, Leonidas J.},
#     Title = {KPConv: Flexible and Deformable Convolution for Point Clouds},
#     Journal = {Proceedings of the IEEE International Conference on Computer Vision},
#     Year = {2019}
# }
# @inproceedings{jaritz2019multi,
# 	title={Multi-view PointNet for 3D Scene Understanding},
# 	author={Jaritz, Maximilian and Gu, Jiayuan and Su, Hao},
# 	booktitle={ICCV Workshop 2019},
# 	year={2019}
# }
# ---------------------------------------------------------------------------------------------------------------------- #
#  Class handling ScanNet dataset.
#  Implements a Dataset, a Sampler, and a collate_fn
#  Upgraded from the original code to load 2D data
# ---------------------------------------------------------------------------------------------------------------------- #

# Common libs
import time
import numpy as np
import pickle
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch
import math
from multiprocessing import Lock
import os.path as osp

# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import grid_subsampling
from utils.config import bcolors

from collections import OrderedDict
from torchvision.transforms import transforms as T
from sklearn.neighbors import NearestNeighbors
from datasets.get_rgbd_overlap_subcloud import compute_rgbd_knn
import glob
import natsort
import open3d as o3d
# ---------------------------------------------------------------------------------------------------------------------- #

# greedily select frames with maximum coverage
def select_frames(rgbd_overlap, num_rgbd_frames):
    selected_frames = []
    # make a copy to avoid modifying input
    rgbd_overlap = rgbd_overlap.copy()
    for i in range(num_rgbd_frames):
        # choose the RGBD frame with the maximum overlap (measured in num basepoints)
        frame_idx = rgbd_overlap.sum(0).argmax()
        selected_frames.append(frame_idx)
        # set all points covered by this frame to invalid
        rgbd_overlap[rgbd_overlap[:, frame_idx]] = False
    return selected_frames

# project depth map to 3D
def depth2xyz(cam_matrix, depth):
    # create xyz coordinates from image position
    v, u = np.indices(depth.shape)
    u, v = u.ravel(), v.ravel()
    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    xyz = (np.linalg.inv(cam_matrix[:3, :3]).dot(uv1_points.T) * depth.ravel()).T
    return xyz

# --------------------------- #
#   Dataset class definition
# --------------------------- #
class ScanNetDataset(PointCloudDataset):
    """Class to handle Scannet dataset."""

    def __init__(self,
                 config,
                 set='training',
                 split = 'train',
                 use_potentials=False,
                 load_data=True,
                 num_rgbd_frames=5,
                 resize=(160, 120),
                 image_normalizer=None,
                 k=3,
                 flip=0.0,
                 color_jitter=None,
                 use_point_color=False,
                 ):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'ScanNet')
        ############
        # Parameters
        ############

        # # Dataset folder path
        # self.path = '/home/dchangyu/mvpnet/ScanNet/'
        # # meta_file path
        # self.META_DIR = '/home/dchangyu/mvpnet/mvpnet/data/meta_files'
        # # cache: pickle files containing point clouds, 3D labels and rgbd overlap
        # self.cache_dir = '/home/dchangyu/mvpnet/ScanNet/cache_rgbd'
        # # includes color, depth, 2D label
        # self.image_dir = '/home/dchangyu/mvpnet/ScanNet/scans_resize_160x120'

        # Dataset folder path
        self.path = config.dataset_path
        # meta_file path
        self.META_DIR = config.META_DIR
        # cache: pickle files containing point clouds, 3D labels and rgbd overlap
        self.cache_dir = config.cache_dir
        # includes color, depth, 2D label
        self.image_dir = config.image_dir


        # exclude some frames with problematic data (e.g. depth frames with zeros everywhere or unreadable labels)
        self.exclude_frames = {
            'scene0243_00': ['1175', '1176', '1177', '1178', '1179', '1180', '1181', '1182', '1183', '1184'],
            'scene0538_00': ['1925', '1928', '1929', '1931', '1932', '1933'],
            'scene0639_00': ['442', '443', '444'],
            'scene0299_01': ['1512'],
        }

        # Training or test set
        self.set = set
        self.split_map = {
            'train': 'scannetv2_train.txt',
            'val': 'scannetv2_val.txt',
            'test': 'scannetv2_test.txt',
        }
        # load split
        self.split = split
        with open(osp.join(self.META_DIR, self.split_map[split]), 'r') as f:
            self.scan_ids = [line.rstrip() for line in f.readlines()]

        # ---------------------------------------------------------------------------- #
        # Build label mapping
        # ---------------------------------------------------------------------------- #

        self.label_id_tsv_path = osp.join(self.META_DIR, 'scannetv2-labels.combined.tsv')
        self.scannet_classes_path = osp.join(self.META_DIR, 'labelids.txt')
        # default ignored value for cross-entropy loss
        self.ignore_value = -100

        # the label in scannet are already nyu40 style
        id_to_class = OrderedDict()
        with open(self.scannet_classes_path, 'r') as f:
            for line in f.readlines():
                class_id, class_name = line.rstrip().split('\t')
                id_to_class[int(class_id)] = class_name
        # get label ids
        self.scannet_mapping = id_to_class
        assert len(self.scannet_mapping) == 20
        # nyu40 -> scannet
        self.nyu40_to_scannet = np.full(shape=41, fill_value=self.ignore_value, dtype=np.int64)
        self.nyu40_to_scannet[list(self.scannet_mapping.keys())] = np.arange(len(self.scannet_mapping))

        # Dict with true label value
        # self.label_to_names = {0: 'unclassified',
        #                        1: 'wall',
        #                        2: 'floor',
        #                        3: 'cabinet',
        #                        4: 'bed',
        #                        5: 'chair',
        #                        6: 'sofa',
        #                        7: 'table',
        #                        8: 'door',
        #                        9: 'window',
        #                        10: 'bookshelf',
        #                        11: 'picture',
        #                        12: 'counter',
        #                        14: 'desk',
        #                        16: 'curtain',
        #                        24: 'refridgerator',
        #                        28: 'shower curtain',
        #                        33: 'toilet',
        #                        34: 'sink',
        #                        36: 'bathtub',
        #                        39: 'otherfurniture'}

        # use index instead of true label value because of confusion matrix (labels=label_value, while network input are index)
        self.label_to_names = {-100: 'unclassified',
                               0: 'wall',
                               1: 'floor',
                               2: 'cabinet',
                               3: 'bed',
                               4: 'chair',
                               5: 'sofa',
                               6: 'table',
                               7: 'door',
                               8: 'window',
                               9: 'bookshelf',
                               10: 'picture',
                               11: 'counter',
                               12: 'desk',
                               13: 'curtain',
                               14: 'refridgerator',
                               15: 'shower curtain',
                               16: 'toilet',
                               17: 'sink',
                               18: 'bathtub',
                               19: 'otherfurniture'}

        # Initialize a bunch of variables concerning class labels
        self.init_labels()

        # List of classes ignored during training
        self.ignored_labels = np.array([-100])

        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes - len(self.ignored_labels)
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        # Number of models used per epoch
        if self.set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        elif self.set in ['validation', 'test']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for ScanNet data: ', self.set)

        # Stop data is not needed
        if not load_data:
            return

        # ---------------------------------------------------------------------------- #
        # 2D parameter
        # ---------------------------------------------------------------------------- #
        # number of selected frames
        self.num_rgbd_frames = num_rgbd_frames
        self.resize = resize
        self.image_normalizer = image_normalizer

        # ---------------------------------------------------------------------------- #
        # 2D-3D parameter
        # ---------------------------------------------------------------------------- #
        # number of neighbors
        self.k = k
        if num_rgbd_frames > 0 and resize:
            depth_size = (640, 480)  # intrinsic matrix is based on 640x480 depth maps.
            self.resize_scale = (depth_size[0] / resize[0], depth_size[1] / resize[1])
        else:
            self.resize_scale = None

        # ---------------------------------------------------------------------------- #
        # 3D parameter
        # ---------------------------------------------------------------------------- #
        # if use 3d color
        self.use_point_color = use_point_color

        # ---------------------------------------------------------------------------- #
        # 2D Augmentation
        # ---------------------------------------------------------------------------- #
        self.flip = flip
        self.color_jitter = T.ColorJitter(*color_jitter) if color_jitter else None

        # ---------------------------------------------------------------------------- #
        # load preprocessed cache data
        # ---------------------------------------------------------------------------- #

        with open(osp.join(self.cache_dir, 'scannetv2_{}.pkl'.format(self.split)), 'rb') as f:
            cache_data = pickle.load(f)
        self.files = cache_data
        self.cloud_names = [scan['scan_id'] for scan in cache_data]

        # sub sampling rate limits
        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        # add 4 more extra containers for 2D data
        self.input_image_xyz = []
        self.input_knn_indices = []
        self.input_images = []
        self.sub_rgbd_dict = []
        #-----------------------------
        self.input_labels = []
        self.pot_trees = []
        self.num_clouds = 0
        self.test_proj = []
        self.validation_labels = []

        # Start loading
        self.load_subsampled_clouds()

        ############################
        # Batch selection parameters
        ############################

        # Initialize value for batch limit (max number of points per batch).
        self.batch_limit = torch.tensor([1], dtype=torch.float32)
        self.batch_limit.share_memory_()

        # Initialize potentials
        if use_potentials:
            self.potentials = []
            self.min_potentials = []
            self.argmin_potentials = []
            for i, tree in enumerate(self.pot_trees):
                self.potentials += [torch.from_numpy(np.random.rand(tree.data.shape[0]) * 1e-3)]
                min_ind = int(torch.argmin(self.potentials[-1]))
                self.argmin_potentials += [min_ind]
                self.min_potentials += [float(self.potentials[-1][min_ind])]

            # Share potential memory
            self.argmin_potentials = torch.from_numpy(np.array(self.argmin_potentials, dtype=np.int64))
            self.min_potentials = torch.from_numpy(np.array(self.min_potentials, dtype=np.float64))
            self.argmin_potentials.share_memory_()
            self.min_potentials.share_memory_()
            for i, _ in enumerate(self.pot_trees):
                self.potentials[i].share_memory_()

            self.worker_waiting = torch.tensor([0 for _ in range(config.input_threads)], dtype=torch.int32)
            self.worker_waiting.share_memory_()
            self.epoch_inds = None
            self.epoch_i = 0

        else:
            raise ValueError("should use potential picking strategy")

        self.worker_lock = Lock()

        return

    # get meta_data path, used in get_rgbd_data function
    def _get_paths(self, scan_dir):
        return {
            'color': osp.join(scan_dir, 'color', '{}.png'),
            # 'color': osp.join(scan_dir, 'color', '{}.jpg'),
            'depth': osp.join(scan_dir, 'depth', '{}.png'),
            'pose': osp.join(scan_dir, 'pose', '{}.txt'),
            'intrinsics_depth': osp.join(scan_dir, 'intrinsic', 'intrinsic_depth.txt'),
        }

    # base on MVPNet code
    def get_rgbd_data(self, sub_rgbd_dict, input_sphere_points, input_sphere_mask, input_sphere_label, whole_scene_points, whole_scene_label):

        scan_id = sub_rgbd_dict['scan_id']
        scan_dir = osp.join(self.image_dir, scan_id)
        paths = self._get_paths(scan_dir)

        # load data
        # nb: number of base points; np: number of all points; ns: number of sphere points
        # nf: number of frames; nbs: number of base points in the sphere
        base_point_ind = sub_rgbd_dict['sub_base_point_ind'].astype(np.int64)  # (nb,)
        base_point_mask = np.zeros_like(input_sphere_mask) # (np,)
        base_point_mask[base_point_ind] = True  # (np,)
        base_point_mask = np.logical_and(base_point_mask,input_sphere_mask)  # (np,)
        pointwise_rgbd_overlap = sub_rgbd_dict['sub_pointwise_rgbd_overlap'].astype(np.bool_)  # (nb, nf)
        chunk_basepoint_rgbd_overlap = pointwise_rgbd_overlap[base_point_mask[base_point_ind]]  # (nbs, nf)
        frame_ids = sub_rgbd_dict['frame_ids']
        cam_matrix = sub_rgbd_dict['cam_matrix'].astype(np.float32)
        # adapt camera matrix
        if self.resize_scale is not None:
            cam_matrix[0] /= self.resize_scale[0]
            cam_matrix[1] /= self.resize_scale[1]

        # greedily choose frames
        selected_frames = select_frames(chunk_basepoint_rgbd_overlap, self.num_rgbd_frames)
        selected_frames = [frame_ids[x] for x in selected_frames]

        # process frames
        image_list = []
        image_xyz_list = []
        image_mask_list = []
        for i, frame_id in enumerate(selected_frames):
            # load image, depth, pose
            image = Image.open(paths['color'].format(frame_id))
            depth = Image.open(paths['depth'].format(frame_id))
            pose = np.loadtxt(paths['pose'].format(frame_id), dtype=np.float32)

            # resize
            if self.resize:
                if not image.size == self.resize:
                    # check if we do not enlarge downsized images
                    assert image.size[0] > self.resize[0] and image.size[1] > self.resize[1]
                    image = image.resize(self.resize, Image.BILINEAR)
                    depth = depth.resize(self.resize, Image.NEAREST)

            # color jitter
            if self.color_jitter is not None:
                image = self.color_jitter(image)

            # normalize image
            image = np.asarray(image, dtype=np.float32) / 255.
            if self.image_normalizer:
                mean, std = self.image_normalizer
                mean = np.asarray(mean, dtype=np.float32)
                std = np.asarray(std, dtype=np.float32)
                image = (image - mean) / std
            image_list.append(image)

            # rescale depth
            depth = np.asarray(depth, dtype=np.float32) / 1000.

            # inverse perspective transformation
            image_xyz = depth2xyz(cam_matrix, depth)  # (h * w, 3)
            # find valid depth
            image_mask = image_xyz[:, 2] > 0  # (h * w)
            # camera -> world
            image_xyz = np.matmul(image_xyz, pose[:3, :3].T) + pose[:3, 3]
            if not np.any(image_mask):
                print('Invalid depth map for frame {} of scan {}.'.format(frame_id, scan_id))

            image_xyz_list.append(image_xyz)
            image_mask_list.append(image_mask)

        # post-process, especially for horizontal flip
        image_ind_list = []
        for i in range(self.num_rgbd_frames):
            h, w, _ = image_list[i].shape
            # reshape
            image_xyz_list[i] = image_xyz_list[i].reshape([h, w, 3])
            image_mask_list[i] = image_mask_list[i].reshape([h, w])
            if self.flip and np.random.rand() < self.flip:
                image_list[i] = np.fliplr(image_list[i])
                image_xyz_list[i] = np.fliplr(image_xyz_list[i])
                image_mask_list[i] = np.fliplr(image_mask_list[i])
            image_mask = image_mask_list[i]
            image_ind = np.nonzero(image_mask.ravel())[0]
            if image_ind.size > 0:
                image_ind_list.append(image_ind + i * h * w)
            else:
                image_ind_list.append([])

        images = np.stack(image_list, axis=0)  # (nv, h, w, 3)
        image_xyz_valid = np.concatenate([image_xyz[image_mask] for image_xyz, image_mask in
                                          zip(image_xyz_list, image_mask_list)], axis=0)
        image_ind_all = np.hstack(image_ind_list)  # (n_valid,)

        # Find k-nn in dense point clouds for each sparse point
        nbrs = NearestNeighbors(n_neighbors=self.k, algorithm='ball_tree').fit(image_xyz_valid) #k (int): k unprojected neighbors of target points
        _, knn_indices = nbrs.kneighbors(input_sphere_points)  # (ns, 3)
        # remap to pixel index
        knn_indices = image_ind_all[knn_indices]

        images = images.astype(np.float32, copy=False), # (nv, h, w, 3)
        image_xyz = np.stack(image_xyz_list, axis=0).astype(np.float32, copy=False),  # (nv, h, w, 3)
        image_mask = np.stack(image_mask_list, axis=0).astype(np.bool_, copy=False),  # (nv, h, w)
        knn_indices = knn_indices.astype(np.int64, copy=False),  # (ns, 3)

        _DEBUG = False
        # visualize basepoints # visualize unprojected point clouds
        # -----------------------------------------------------------------------------------------------
        # if _DEBUG == True:
            # from mvpnet.utils.o3d_util import draw_point_cloud
            # from mvpnet.utils.visualize import label2color
            # pts_vis = draw_point_cloud(input_sphere_points, label2color(input_sphere_label, style='scannet'))
            # base_pts_vis = draw_point_cloud(whole_scene_points[base_point_mask], [1., 0., 0.])
            # pcd_xyz = draw_point_cloud(image_xyz_valid, [0., 1., 0.])
            # pcd_scene = draw_point_cloud(whole_scene_points, label2color(whole_scene_label, style='scannet'))
            # path = '/home/dchangyu/mvpnet/'
            # o3d.io.write_point_cloud(path + scan_id + 'input_sphere_points' + '.ply', pts_vis)
            # o3d.io.write_point_cloud(path + scan_id + 'base_points' + '.ply', base_pts_vis)
            # o3d.io.write_point_cloud(path + scan_id + 'image_xyz_points' + '.ply', pcd_xyz)
            # o3d.io.write_point_cloud(path + scan_id + 'whole_scene_points' + '.ply', pcd_scene)
        # -----------------------------------------------------------------------------------------------
        return images, image_xyz, knn_indices, image_mask


    def __len__(self):
        """
        Return the length of data here
        """
        return len(self.cloud_names)

    def __getitem__(self, batch_i):
        """
        The main thread gives a list of indices to load a batch. Each worker is going to work in parallel to load a
        different list of indices.
        """

        if self.use_potentials:
            return self.potential_item(batch_i)
        else:
            raise ValueError("should use potential")

    def potential_item(self, batch_i, debug_workers=False):

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []

        xyz_and_color_list = []
        z_list = []
        p_f_list = []
        image_xyz_list = []
        images_list = []
        knn_list = []

        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0

        info = get_worker_info()
        if info is not None:
            wid = info.id
        else:
            wid = None

        while True:

            t += [time.time()]

            if debug_workers:
                message = ''
                for wi in range(info.num_workers):
                    if wi == wid:
                        message += ' {:}X{:} '.format(bcolors.FAIL, bcolors.ENDC)
                    elif self.worker_waiting[wi] == 0:
                        message += '   '
                    elif self.worker_waiting[wi] == 1:
                        message += ' | '
                    elif self.worker_waiting[wi] == 2:
                        message += ' o '
                print(message)
                self.worker_waiting[wid] = 0

            with self.worker_lock:

                if debug_workers:
                    message = ''
                    for wi in range(info.num_workers):
                        if wi == wid:
                            message += ' {:}v{:} '.format(bcolors.OKGREEN, bcolors.ENDC)
                        elif self.worker_waiting[wi] == 0:
                            message += '   '
                        elif self.worker_waiting[wi] == 1:
                            message += ' | '
                        elif self.worker_waiting[wi] == 2:
                            message += ' o '
                    print(message)
                    self.worker_waiting[wid] = 1

                # Get potential minimum
                cloud_ind = int(torch.argmin(self.min_potentials))
                point_ind = int(self.argmin_potentials[cloud_ind])

                # check if they are same scene
                check_scan_id = self.cloud_names[cloud_ind]
                assert check_scan_id == self.sub_rgbd_dict[cloud_ind]['scan_id'], 'Mismatch scan_id: {} vs {}.'.format(check_scan_id, self.sub_rgbd_dict[cloud_ind]['scan_id'])

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Center point of input region
                center_point = pot_points[point_ind, :].reshape(1, -1)

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]
                # Update potentials (Tukey weights)
                tukeys = np.square(1 - d2s / np.square(self.config.in_radius))
                tukeys[d2s > np.square(self.config.in_radius)] = 0
                #self.potentials[cloud_ind][pot_inds] += tukeys
                self.potentials[cloud_ind][pot_inds] += torch.from_numpy(tukeys)
                min_ind = torch.argmin(self.potentials[cloud_ind])
                self.min_potentials[[cloud_ind]] = self.potentials[cloud_ind][min_ind]
                self.argmin_potentials[[cloud_ind]] = min_ind

            t += [time.time()]

            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            # Indices of sphere mask for input region, should slightly bigger then input sphere
            input_sphere_mask_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                              r=self.config.in_radius + 0.1)[0]

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]

            # Collect labels and colors, Note that the coordinates of the input point are the distance from the center of the ball
            input_points = (points[input_inds] - center_point).astype(np.float32)
            # 3d colors
            input_colors = self.input_colors[cloud_ind][input_inds]

            # 把feat_aggre 所需的points以feature形式输入, get input sphere
            input_points_feat_aggre = points[input_inds].astype(np.float32)

            if self.set in ['test']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                # input are label indices
                input_labels = self.input_labels[cloud_ind][input_inds].astype(np.int64)

            whole_scene_label = self.input_labels[cloud_ind]

            input_points_inds = np.zeros(points.shape[0], dtype=bool)
            input_points_inds[input_sphere_mask_inds] = True
            input_sphere_mask = input_points_inds

            # ---------------------------------------------------------------------------- #
            # On-the-fly find frames maximally cover the sphere
            # ---------------------------------------------------------------------------- #

            if self.num_rgbd_frames > 0:
                images, image_xyz, knn_indices, image_mask = self.get_rgbd_data(self.sub_rgbd_dict[cloud_ind],
                                                                                input_points_feat_aggre,
                                                                                input_sphere_mask,
                                                                                input_labels,
                                                                                points,
                                                                                whole_scene_label,
                                                                                )
            else:
                raise ValueError("should select at least one image")
            # ---------------------------------------------------------------------------- #
            # original output are tuple
            # images (nv, h, w, 3)
            # image_xyz (nv, h, w, 3)
            # knn (np, 3) np can be different
            # image_mask (nv, h, w)
            # ---------------------------------------------------------------------------- #
            # expand batch dims
            # ---------------------------------------------------------------------------- #
            images = np.moveaxis(images[0], -1, 1) # (nv, 3, h, w)
            images = np.expand_dims(images, axis=0)  # (1, nv, 3, h, w)
            image_xyz = np.expand_dims(image_xyz[0], axis=0) #(1, nv, h, w, 3)
            knn_indices = np.expand_dims(knn_indices[0], axis=0) # (1, np , k)
            input_points_feat_aggre = np.expand_dims(input_points_feat_aggre, axis=0) # (1, np ,3)
            # ---------------------------------------------------------------------------- #

            t += [time.time()]

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            if np.random.rand() > self.config.augment_color:
                input_colors *= 0

            # Get original height as additional feature, rgb+z
            input_z_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            # Get original xyz as additional feature
            # input_xyz_features = np.hstack((input_points[:, :] + center_point[:, :])).astype(np.float32)

            # 3+3=6, stack xyz with RGB colors: (np,6)
            input_3d_features = np.hstack((input_colors, input_points[:, :] + center_point[:, :])).astype(np.float32)

            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            xyz_and_color_list += [input_3d_features]
            z_list += [input_z_features]
            p_f_list += [input_points_feat_aggre]
            image_xyz_list += [image_xyz]
            images_list += [images]
            knn_list += [knn_indices]
            l_list += [input_labels]
            pi_list += [input_inds]
            i_list += [point_ind]
            ci_list += [cloud_ind]
            s_list += [scale]
            R_list += [R]

            # Update batch size
            batch_n += n

            # In case batch is full, stop
            if batch_n > int(self.batch_limit):
                break

            # Randomly drop some points (act as an augmentation process and a safety for GPU memory consumption)
            # if n > int(self.batch_limit):
            #    input_inds = np.random.choice(input_inds, size=int(self.batch_limit) - 1, replace=False)
            #    n = input_inds.shape[0]

        ###################
        # Concatenate batch
        ###################
        # All dimensions should be the same except for the axis position
        stacked_points = np.concatenate(p_list, axis=0) #(np , 3)
        xyz_and_color_features = np.concatenate(xyz_and_color_list, axis=0) #(np,6)
        input_height_features = np.concatenate(z_list, axis=0)  # (np,1)
        feat_aggre_points = np.concatenate(p_f_list, axis=1) #(1, np , 3)
        stacked_image_xyz = np.concatenate(image_xyz_list, axis=0) #(1, nv, h, w, 3)
        stacked_images = np.concatenate(images_list, axis=0)  #(1, nv, 3, h, w)
        # stacked_knn_indices = np.concatenate(knn_list,axis=1) #(1, np , 3)
        #-------------------------------------------------
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # ---------------------------------------------------------------------------- #
        # Input features
        # ---------------------------------------------------------------------------- #
        # stack the features with constant 1 to insure black/dark points are not ignored
        stacked_3D_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)

        ### For late fusion variant ####
        if self.config.late_fusion == True:
            # without xyz and color
            if self.config.in_features_dim == 1:
                pass
                # 1 + Z
            elif self.config.in_features_dim == 2:
                stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features[:, 5:]))

            elif self.config.in_features_dim == 4:
                # 1 + 3 rgb channels
                if self.use_point_color == True:
                    stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features[:, :3]))
                # 1 + 3 xyz channels
                else:
                    stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features[:, 3:]))
            # 1 + 3 rgb channels + 3 XYZ channels
            elif self.config.in_features_dim == 7:
                stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features))

        ### For middle fusion variant ####
        elif self.config.middle_fusion == True:
            # without xyz and color
            if self.config.in_features_dim_3d == 1:
                pass
                # 1 + Z
            elif self.config.in_features_dim_3d == 2:
                stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features[:, 5:]))

            elif self.config.in_features_dim_3d == 4:
                # 1 + 3 rgb channels
                if self.use_point_color == True:
                    stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features[:, :3]))
                # 1 + 3 xyz channels
                else:
                    stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features[:, 3:]))
            # 1 + 3 rgb channels + 3 XYZ channels
            elif self.config.in_features_dim_3d == 7:
                stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features))

        ### For Early fusion variant ####
        # 64 image feature channels +1
        elif self.config.early_fusion == True:
            if self.config.in_features_dim == 65:
                pass
            # 64+z+1
            elif self.config.in_features_dim == 66:
                stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features[:, 5:]))

            elif self.config.in_features_dim == 68:
                # 64 + 3 color channels + 1
                if self.use_point_color == True:
                    stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features[:, :3]))
                # 64 + 3 xyz channels + 1
                else:
                    stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features[:, 3:]))

            # 64 + 3 color channels + z + 1
            elif self.config.in_features_dim == 69:
                stacked_3D_features = np.hstack((stacked_3D_features, input_height_features))
            # 64+ 3 color channels + 3 XYZ channels + 1
            elif self.config.in_features_dim == 71:
                stacked_3D_features = np.hstack((stacked_3D_features, xyz_and_color_features))

            else:
                raise ValueError('choose one fusion!')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs_sphere(stacked_points,
                                                    stacked_image_xyz,
                                                    stacked_images,
                                                    feat_aggre_points,
                                                    labels,
                                                    stack_lengths)

        t += [time.time()]

        # Add scale and rotation for testing, add knn_list to do feature lifting scene-wise
        input_list += [scales, rots, cloud_inds, point_inds, input_inds, knn_list, stacked_3D_features]

        if debug_workers:
            message = ''
            for wi in range(info.num_workers):
                if wi == wid:
                    message += ' {:}0{:} '.format(bcolors.OKBLUE, bcolors.ENDC)
                elif self.worker_waiting[wi] == 0:
                    message += '   '
                elif self.worker_waiting[wi] == 1:
                    message += ' | '
                elif self.worker_waiting[wi] == 2:
                    message += ' o '
            print(message)
            self.worker_waiting[wid] = 2

        t += [time.time()]

        # Display timings
        debugT = False
        if debugT:
            print('\n************************\n')
            print('Timings:')
            ti = 0
            N = 5
            mess = 'Init ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Pots ...... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Sphere .... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Collect ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += 1
            mess = 'Augment ... {:5.1f}ms /'
            loop_times = [1000 * (t[ti + N * i + 1] - t[ti + N * i]) for i in range(len(stack_lengths))]
            for dt in loop_times:
                mess += ' {:5.1f}'.format(dt)
            print(mess.format(np.sum(loop_times)))
            ti += N * (len(stack_lengths) - 1) + 1
            print('concat .... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('input ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('stack ..... {:5.1f}ms'.format(1000 * (t[ti+1] - t[ti])))
            ti += 1
            print('\n************************\n')
        return input_list



    def load_subsampled_clouds(self):

        # Parameter
        dl = self.config.first_subsampling_dl

        # Create path for files
        tree_path = join(self.path, 'input_{:.3f}'.format(dl))
        if not exists(tree_path):
            makedirs(tree_path)

        ##############
        # Load KDTrees
        ##############

        for index, data_i in enumerate(self.files):

            # Restart timer
            t0 = time.time()

            # Get cloud name
            cloud_name = self.cloud_names[index]
            assert cloud_name == self.cloud_names[index], 'Mismatch scan_id: {} vs {}.'.format(cloud_name, self.cloud_names[index])

            # Name of the input files
            KDTree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_pkl_file = join(tree_path, '{:s}.pkl'.format(cloud_name))

            # Check if inputs have already been computed
            if exists(KDTree_file):
                print('\nFound KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # read ply with data
                with open(sub_pkl_file, 'rb') as f:
                    data_load = pickle.load(f)

                sub_colors = data_load['sub_colors']
                sub_labels = data_load['sub_labels']
                sub_rgbd_dict = data_load['rgbd_dict']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # ---------------------------------------------------------------------------- #
                # Load 3D data
                # ---------------------------------------------------------------------------- #
                points = data_i['points'].astype(np.float32) #(n,3)
                #colors = np.vstack((data['red'], data['green'], data['blue'])).T #(n,3)
                colors = data_i['colors'] #(n,3) # uint8
                labels = data_i['seg_label'] #int64?
                # map labels to its ids: -100->0, 0->1, ...., 19 -> 20
                labels = self.nyu40_to_scannet[labels].astype(np.int32)

                # Subsample cloud
                sub_points, sub_colors, sub_labels = grid_subsampling(points,
                                                          features=colors,
                                                          labels=labels,
                                                          sampleDl=dl)

                # Rescale float color and squeeze label
                sub_colors = sub_colors / 255 # float32
                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)
                # search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
                # search_tree.fit(sub_points)

                # ---------------------------------------------------------------------------- #
                # Compute rgbd overlap for subsampled point cloud
                # ---------------------------------------------------------------------------- #

                # set random seed, there is random choice in compute rgbd overlap
                seed = int(cloud_name[5:9] + cloud_name[10:12])
                # print(seed)
                np.random.seed(seed)

                scan_dir = osp.join(self.image_dir, cloud_name)

                paths = self._get_paths(scan_dir)

                glob_path = osp.join(scan_dir, 'color', '*')
                cam_matrix = np.loadtxt(paths['intrinsics_depth'], dtype=np.float32)
                color_paths = natsort.natsorted(glob.glob(glob_path))
                exclude_ids = self.exclude_frames.get(cloud_name, [])
                frame_ids = [osp.splitext(osp.basename(x))[0] for x in color_paths]
                frame_ids = [x for x in frame_ids if x not in exclude_ids]
                if not frame_ids:
                    print('WARNING: No frames found, check glob path {}'.format(glob_path))

                base_point_ind, pointwise_rgbd_overlap = compute_rgbd_knn(cloud_name, frame_ids, cam_matrix, paths, sub_points)  # cache_data return whole scence points

                sub_rgbd_dict = {'scan_id' : cloud_name}
                sub_rgbd_dict.update({
                    'sub_base_point_ind': base_point_ind,
                    'sub_pointwise_rgbd_overlap': pointwise_rgbd_overlap,
                    'frame_ids': frame_ids,
                    'cam_matrix': cam_matrix,
                })

                #-----------------------------------------------------------------------------#
                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save pkl save points feature and labels
                sub_data = {'sub_points': sub_points, 'sub_labels' : sub_labels, 'sub_colors' : sub_colors,'rgbd_dict' : sub_rgbd_dict}
                with open(sub_pkl_file, 'wb') as f:
                    pickle.dump(sub_data, f)

            # Fill data containers
            self.input_trees += [search_tree]
            self.input_colors += [sub_colors]
            self.input_labels += [sub_labels]
            # ---------------
            self.sub_rgbd_dict += [sub_rgbd_dict]


            size1 = sub_labels.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size1 * 1e-6, time.time() - t0))
            # size2 = image_xyz.shape[0] * 4 * 7
            # print('{:.1f} MB loaded in {:.1f}s'.format(size2 * 1e-6, time.time() - t0))
            # size3 = images.shape[0] * 4 * 7
            # print('{:.1f} MB loaded in {:.1f}s'.format(size3 * 1e-6, time.time() - t0))

        ############################################
        # Coarse potential locations of input sphere
        ############################################

        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            # pot_dl = self.config.in_radius
            cloud_ind = 0

            for index, data_i in enumerate(self.files):

                # Get cloud name
                cloud_name = self.cloud_names[index]

                # Name of the input files
                coarse_KDTree_file = join(tree_path, '{:s}_coarse_KDTree.pkl'.format(cloud_name))

                # Check if inputs have already been computed
                if exists(coarse_KDTree_file):
                    # Read pkl with search tree
                    with open(coarse_KDTree_file, 'rb') as f:
                        search_tree = pickle.load(f)
                else:

                    # Subsample cloud
                    sub_points = np.array(self.input_trees[cloud_ind].data, copy=False)
                    coarse_points = grid_subsampling(sub_points.astype(np.float32), sampleDl=pot_dl)

                    # Get chosen neighborhoods
                    search_tree = KDTree(coarse_points, leaf_size=10)

                    # Save KDTree
                    with open(coarse_KDTree_file, 'wb') as f:
                        pickle.dump(search_tree, f)

                # Fill data containers
                self.pot_trees += [search_tree]
                cloud_ind += 1

            print('Done in {:.1f}s'.format(time.time() - t0))

        ######################
        # Reprojection indices
        ######################
        # Only necessary for validation and test sets
        # Get number of clouds
        self.num_clouds = len(self.input_trees)

        if self.set in ['validation', 'test']:
            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            for index, data_i in enumerate(self.files):

                # Restart timer
                t0 = time.time()

                # Get info on this cloud
                cloud_name = self.cloud_names[index]

                # File name for saving
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))

                # Try to load previous indices
                if exists(proj_file):
                    with open(proj_file, 'rb') as f:
                        proj_inds, labels = pickle.load(f)
                else:

                    points = data_i['points'].astype(np.float32)
                    labels = data_i['seg_label']
                    # map labels to its ids: -100->0, 0->1, ...., 19 -> 20
                    labels = self.nyu40_to_scannet[labels].astype(np.int32)
                    # Compute projection inds
                    idxs = self.input_trees[index].query(points, return_distance=False)
                    #dists, idxs = self.input_trees[i_cloud].kneighbors(points)
                    proj_inds = np.squeeze(idxs).astype(np.int32)

                    # Save
                    with open(proj_file, 'wb') as f:
                        pickle.dump([proj_inds, labels], f)

                self.test_proj += [proj_inds]
                self.validation_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

        print()
        return

    def load_evaluation_points(self, file_path):
        """
        Load points (from test or validation split) on which the metrics should be evaluated
        """

        # Get original points
        # data = read_ply(file_path)
        # return np.vstack((data['x'], data['y'], data['z'])).T #(n,3)
        # file_path is passed in the individual dictionaries in the pkl file list
        return file_path['points'].astype(dtype=np.float32, copy=False) #(n,3)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility classes definition
#       \********************************/


class ScanNetSampler(Sampler):
    """Sampler for S3DIS"""

    def __init__(self, dataset: ScanNetDataset):
        Sampler.__init__(self, dataset)

        # Dataset used by the sampler (no copy is made in memory)
        self.dataset = dataset

        # Number of step per epoch
        if dataset.set == 'training':
            self.N = dataset.config.epoch_steps
        else:
            self.N = dataset.config.validation_size

        return

    def __iter__(self):
        """
        Yield next batch indices here. In this dataset, this is a dummy sampler that yield the index of batch element
        (input sphere) in epoch instead of the list of point indices
        """

        if not self.dataset.use_potentials:

            # Initiate current epoch ind
            self.dataset.epoch_i *= 0
            self.dataset.epoch_inds *= 0

            # Initiate container for indices
            all_epoch_inds = np.zeros((2, 0), dtype=np.int32)

            # Number of sphere centers taken per class in each cloud
            num_centers = self.N * self.dataset.config.batch_num
            random_pick_n = int(np.ceil(num_centers / (self.dataset.num_clouds * self.dataset.config.num_classes)))

            # Choose random points of each class for each cloud
            for cloud_ind, cloud_labels in enumerate(self.dataset.input_labels):
                epoch_indices = np.empty((0,), dtype=np.int32)
                for label_ind, label in enumerate(self.dataset.label_values):
                    if label not in self.dataset.ignored_labels:
                        label_indices = np.where(np.equal(cloud_labels, label))[0]
                        if len(label_indices) <= random_pick_n:
                            epoch_indices = np.hstack((epoch_indices, label_indices))
                        elif len(label_indices) < 50 * random_pick_n:
                            new_randoms = np.random.choice(label_indices, size=random_pick_n, replace=False)
                            epoch_indices = np.hstack((epoch_indices, new_randoms.astype(np.int32)))
                        else:
                            rand_inds = []
                            while len(rand_inds) < random_pick_n:
                                rand_inds = np.unique(np.random.choice(label_indices, size=5 * random_pick_n, replace=True))
                            epoch_indices = np.hstack((epoch_indices, rand_inds[:random_pick_n].astype(np.int32)))

                # Stack those indices with the cloud index
                epoch_indices = np.vstack((np.full(epoch_indices.shape, cloud_ind, dtype=np.int32), epoch_indices))

                # Update the global indice container
                all_epoch_inds = np.hstack((all_epoch_inds, epoch_indices))

            # Random permutation of the indices
            random_order = np.random.permutation(all_epoch_inds.shape[1])
            all_epoch_inds = all_epoch_inds[:, random_order].astype(np.int64)

            # Update epoch inds
            self.dataset.epoch_inds += torch.from_numpy(all_epoch_inds[:, :num_centers])

        # Generator loop
        for i in range(self.N):
            yield i

    def __len__(self):
        """
        The number of yielded samples is variable
        """
        return self.N

    def fast_calib(self):
        """
        This method calibrates the batch sizes while ensuring the potentials are well initialized. Indeed on a dataset
        , before potential have been updated over the dataset, there are cahnces that all the dense area
        are picked in the begining and in the end, we will have very large batch of small point clouds
        :return:
        """

        # Estimated average batch size and target value
        estim_b = 0
        target_b = self.dataset.config.batch_num

        # Calibration parameters
        low_pass_T = 10
        Kp = 100.0
        finer = False
        breaking = False

        # Convergence parameters
        smooth_errors = []
        converge_threshold = 0.1

        t = [time.time()]
        last_display = time.time()
        mean_dt = np.zeros(2)

        for epoch in range(10):
            for i, test in enumerate(self):

                # New time
                t = t[-1:]
                t += [time.time()]

                # batch length
                b = len(test)

                # Update estim_b (low pass filter)
                estim_b += (b - estim_b) / low_pass_T

                # Estimate error (noisy)
                error = target_b - b

                # Save smooth errors for convergene check
                smooth_errors.append(target_b - estim_b)
                if len(smooth_errors) > 10:
                    smooth_errors = smooth_errors[1:]

                # Update batch limit with P controller
                self.dataset.batch_limit += Kp * error

                # finer low pass filter when closing in
                if not finer and np.abs(estim_b - target_b) < 1:
                    low_pass_T = 100
                    finer = True

                # Convergence
                if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                    breaking = True
                    break

                # Average timing
                t += [time.time()]
                mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d},  //  {:.1f}ms {:.1f}ms'
                    print(message.format(i,
                                         estim_b,
                                         int(self.dataset.batch_limit),
                                         1000 * mean_dt[0],
                                         1000 * mean_dt[1]))

            if breaking:
                break

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=True, force_redo=False):
        """
        Method performing batch and neighbors calibration.
            Batch calibration: Set "batch_limit" (the maximum number of points allowed in every batch) so that the
                               average batch size (number of stacked pointclouds) is the one asked.
        Neighbors calibration: Set the "neighborhood_limits" (the maximum number of neighbors allowed in convolutions)
                               so that 90% of the neighborhoods remain untouched. There is a limit for each layer.
        """

        ##############################
        # Previously saved calibration
        ##############################

        print('\nStarting Calibration (use verbose=True for more details)')
        t0 = time.time()

        redo = force_redo

        # Batch limit
        # ***********

        # Load batch_limit dictionary
        batch_lim_file = join(self.dataset.path, 'batch_limits.pkl')
        if exists(batch_lim_file):
            with open(batch_lim_file, 'rb') as file:
                batch_lim_dict = pickle.load(file)
        else:
            batch_lim_dict = {}

        # Check if the batch limit associated with current parameters exists
        if self.dataset.use_potentials:
            sampler_method = 'potentials'
        else:
            sampler_method = 'random'
        key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                               self.dataset.config.in_radius,
                                               self.dataset.config.first_subsampling_dl,
                                               self.dataset.config.batch_num)
        if not redo and key in batch_lim_dict:
            self.dataset.batch_limit[0] = batch_lim_dict[key]
        else:
            redo = True

        if verbose:
            print('\nPrevious calibration found:')
            print('Check batch limit dictionary')
            if key in batch_lim_dict:
                color = bcolors.OKGREEN
                v = str(int(batch_lim_dict[key]))
            else:
                color = bcolors.FAIL
                v = '?'
            print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        # Neighbors limit
        # ***************

        # Load neighb_limits dictionary
        neighb_lim_file = join(self.dataset.path, 'neighbors_limits.pkl')
        if exists(neighb_lim_file):
            with open(neighb_lim_file, 'rb') as file:
                neighb_lim_dict = pickle.load(file)
        else:
            neighb_lim_dict = {}

        # Check if the limit associated with current parameters exists (for each layer)
        neighb_limits = []
        for layer_ind in range(self.dataset.config.num_layers):

            dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
            if self.dataset.config.deform_layers[layer_ind]:
                r = dl * self.dataset.config.deform_radius
            else:
                r = dl * self.dataset.config.conv_radius

            key = '{:.3f}_{:.3f}'.format(dl, r)
            if key in neighb_lim_dict:
                neighb_limits += [neighb_lim_dict[key]]

        if not redo and len(neighb_limits) == self.dataset.config.num_layers:
            self.dataset.neighborhood_limits = neighb_limits
        else:
            redo = True

        if verbose:
            print('Check neighbors limit dictionary')
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2**layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)

                if key in neighb_lim_dict:
                    color = bcolors.OKGREEN
                    v = str(neighb_lim_dict[key])
                else:
                    color = bcolors.FAIL
                    v = '?'
                print('{:}\"{:s}\": {:s}{:}'.format(color, key, v, bcolors.ENDC))

        if redo:

            ############################
            # Neighbors calib parameters
            ############################

            # From config parameter, compute higher bound of neighbors number in a neighborhood
            hist_n = int(np.ceil(4 / 3 * np.pi * (self.dataset.config.deform_radius + 1) ** 3))

            # Histogram of neighborhood sizes
            neighb_hists = np.zeros((self.dataset.config.num_layers, hist_n), dtype=np.int32)

            ########################
            # Batch calib parameters
            ########################

            # Estimated average batch size and target value
            estim_b = 0
            target_b = self.dataset.config.batch_num

            # Calibration parameters
            low_pass_T = 10
            Kp = 100.0
            finer = False

            # Convergence parameters
            smooth_errors = []
            converge_threshold = 0.1

            # Loop parameters
            last_display = time.time()
            i = 0
            breaking = False

            #####################
            # Perform calibration
            #####################

            for epoch in range(10):
                for batch_i, batch in enumerate(dataloader):

                    # Update neighborhood histogram
                    counts = [np.sum(neighb_mat.numpy() < neighb_mat.shape[0], axis=1) for neighb_mat in batch.neighbors]
                    hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
                    neighb_hists += np.vstack(hists)

                    # batch length
                    b = len(batch.cloud_inds)

                    # Update estim_b (low pass filter)
                    estim_b += (b - estim_b) / low_pass_T

                    # Estimate error (noisy)
                    error = target_b - b

                    # Save smooth errors for convergene check
                    smooth_errors.append(target_b - estim_b)
                    if len(smooth_errors) > 10:
                        smooth_errors = smooth_errors[1:]

                    # Update batch limit with P controller
                    self.dataset.batch_limit += Kp * error

                    # finer low pass filter when closing in
                    if not finer and np.abs(estim_b - target_b) < 1:
                        low_pass_T = 100
                        finer = True

                    # Convergence
                    if finer and np.max(np.abs(smooth_errors)) < converge_threshold:
                        breaking = True
                        break

                    i += 1
                    t = time.time()

                    # Console display (only one per second)
                    if verbose and (t - last_display) > 1.0:
                        last_display = t
                        message = 'Step {:5d}  estim_b ={:5.2f} batch_limit ={:7d}'
                        print(message.format(i,
                                             estim_b,
                                             int(self.dataset.batch_limit)))

                if breaking:
                    break

            # Use collected neighbor histogram to get neighbors limit
            cumsum = np.cumsum(neighb_hists.T, axis=0)
            percentiles = np.sum(cumsum < (untouched_ratio * cumsum[hist_n - 1, :]), axis=0)
            self.dataset.neighborhood_limits = percentiles

            if verbose:

                # Crop histogram
                while np.sum(neighb_hists[:, -1]) == 0:
                    neighb_hists = neighb_hists[:, :-1]
                hist_n = neighb_hists.shape[1]

                print('\n**************************************************\n')
                line0 = 'neighbors_num '
                for layer in range(neighb_hists.shape[0]):
                    line0 += '|  layer {:2d}  '.format(layer)
                print(line0)
                for neighb_size in range(hist_n):
                    line0 = '     {:4d}     '.format(neighb_size)
                    for layer in range(neighb_hists.shape[0]):
                        if neighb_size > percentiles[layer]:
                            color = bcolors.FAIL
                        else:
                            color = bcolors.OKGREEN
                        line0 += '|{:}{:10d}{:}  '.format(color,
                                                         neighb_hists[layer, neighb_size],
                                                         bcolors.ENDC)

                    print(line0)

                print('\n**************************************************\n')
                print('\nchosen neighbors limits: ', percentiles)
                print()

            # Save batch_limit dictionary
            if self.dataset.use_potentials:
                sampler_method = 'potentials'
            else:
                sampler_method = 'random'
            key = '{:s}_{:.3f}_{:.3f}_{:d}'.format(sampler_method,
                                                   self.dataset.config.in_radius,
                                                   self.dataset.config.first_subsampling_dl,
                                                   self.dataset.config.batch_num)
            batch_lim_dict[key] = float(self.dataset.batch_limit)
            with open(batch_lim_file, 'wb') as file:
                pickle.dump(batch_lim_dict, file)

            # Save neighb_limit dictionary
            for layer_ind in range(self.dataset.config.num_layers):
                dl = self.dataset.config.first_subsampling_dl * (2 ** layer_ind)
                if self.dataset.config.deform_layers[layer_ind]:
                    r = dl * self.dataset.config.deform_radius
                else:
                    r = dl * self.dataset.config.conv_radius
                key = '{:.3f}_{:.3f}'.format(dl, r)
                neighb_lim_dict[key] = self.dataset.neighborhood_limits[layer_ind]
            with open(neighb_lim_file, 'wb') as file:
                pickle.dump(neighb_lim_dict, file)


        print('Calibration done in {:.1f}s\n'.format(time.time() - t0))
        return


class ScanNetCustomBatch:
    """Custom batch definition with memory pinning for S3DIS"""

    def __init__(self, input_list):

        # Get rid of batch dimension
        input_list = input_list[0]

        # Number of layers
        # The interpretation of the two constants in https://github.com/HuguesTHOMAS/KPConv-PyTorch/issues/69
        L = (len(input_list) - 11) // 5

        # Extract input tensors from the list of numpy array
        ind = 0
        self.points = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.neighbors = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.pools = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.upsamples = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.lengths = [torch.from_numpy(nparray) for nparray in input_list[ind:ind+L]]
        ind += L
        self.feat_aggre_points = torch.from_numpy(input_list[ind])
        ind += 1
        self.image_xyz = torch.from_numpy(input_list[ind])
        ind += 1
        self.images = torch.from_numpy(input_list[ind])
        #--------------------------------------------------------------------------
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.knn_list = input_list[ind]
        ind += 1
        self.feature_3d = torch.from_numpy(input_list[ind])

        return

    def pin_memory(self):
        """
        Manual pinning of the memory
        """

        self.points = [in_tensor.pin_memory() for in_tensor in self.points]
        self.neighbors = [in_tensor.pin_memory() for in_tensor in self.neighbors]
        self.pools = [in_tensor.pin_memory() for in_tensor in self.pools]
        self.upsamples = [in_tensor.pin_memory() for in_tensor in self.upsamples]
        self.lengths = [in_tensor.pin_memory() for in_tensor in self.lengths]
        self.feat_aggre_points = self.feat_aggre_points.pin_memory()
        self.image_xyz = self.image_xyz.pin_memory()
        self.images = self.images.pin_memory()

        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()
        self.knn_list = self.knn_list
        self.feature_3d = self.feature_3d.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        self.feat_aggre_points = self.feat_aggre_points.to(device)
        self.image_xyz = self.image_xyz.to(device)
        self.images = self.images.to(device)

        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)
        self.knn_list = self.knn_list
        self.feature_3d = self.feature_3d.to(device)

        return self

    def unstack_points(self, layer=None):
        """Unstack the points"""
        return self.unstack_elements('points', layer)

    def unstack_neighbors(self, layer=None):
        """Unstack the neighbors indices"""
        return self.unstack_elements('neighbors', layer)

    def unstack_pools(self, layer=None):
        """Unstack the pooling indices"""
        return self.unstack_elements('pools', layer)

    def unstack_elements(self, element_name, layer=None, to_numpy=True):
        """
        Return a list of the stacked elements in the batch at a certain layer. If no layer is given, then return all
        layers
        """

        if element_name == 'points':
            elements = self.points
        elif element_name == 'neighbors':
            elements = self.neighbors
        elif element_name == 'pools':
            elements = self.pools[:-1]
        else:
            raise ValueError('Unknown element name: {:s}'.format(element_name))

        all_p_list = []
        for layer_i, layer_elems in enumerate(elements):

            if layer is None or layer == layer_i:

                i0 = 0
                p_list = []
                if element_name == 'pools':
                    lengths = self.lengths[layer_i+1]
                else:
                    lengths = self.lengths[layer_i]

                for b_i, length in enumerate(lengths):

                    elem = layer_elems[i0:i0 + length]
                    if element_name == 'neighbors':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= i0
                    elif element_name == 'pools':
                        elem[elem >= self.points[layer_i].shape[0]] = -1
                        elem[elem >= 0] -= torch.sum(self.lengths[layer_i][:b_i])
                    i0 += length

                    if to_numpy:
                        p_list.append(elem.numpy())
                    else:
                        p_list.append(elem)

                if layer == layer_i:
                    return p_list

                all_p_list.append(p_list)

        return all_p_list


def ScanNetCollate(batch_data):
    return ScanNetCustomBatch(batch_data)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Debug functions
#       \*********************/


def debug_upsampling(dataset, loader):
    """Shows which labels are sampled according to strategy chosen"""


    for epoch in range(10):

        for batch_i, batch in enumerate(loader):

            pc1 = batch.points[1].numpy()
            pc2 = batch.points[2].numpy()
            up1 = batch.upsamples[1].numpy()

            print(pc1.shape, '=>', pc2.shape)
            print(up1.shape, np.max(up1))

            pc2 = np.vstack((pc2, np.zeros_like(pc2[:1, :])))

            # Get neighbors distance
            p0 = pc1[10, :]
            neighbs0 = up1[10, :]
            neighbs0 = pc2[neighbs0, :] - p0
            d2 = np.sum(neighbs0 ** 2, axis=1)

            print(neighbs0.shape)
            print(neighbs0[:5])
            print(d2[:5])

            print('******************')
        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_timing(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)
    estim_b = dataset.config.batch_num
    estim_N = 0

    for epoch in range(10):

        for batch_i, batch in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Update estim_b (low pass filter)
            estim_b += (len(batch.cloud_inds) - estim_b) / 100
            estim_N += (batch.features.shape[0] - estim_N) / 10

            # Pause simulating computations
            time.sleep(0.05)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > -1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> (ms/batch) {:8.2f} {:8.2f} / batch = {:.2f} - {:.0f}'
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1],
                                     estim_b,
                                     estim_N))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_show_clouds(dataset, loader):


    for epoch in range(10):

        clouds = []
        cloud_normals = []
        cloud_labels = []

        L = dataset.config.num_layers

        for batch_i, batch in enumerate(loader):

            # Print characteristics of input tensors
            print('\nPoints tensors')
            for i in range(L):
                print(batch.points[i].dtype, batch.points[i].shape)
            print('\nNeigbors tensors')
            for i in range(L):
                print(batch.neighbors[i].dtype, batch.neighbors[i].shape)
            print('\nPools tensors')
            for i in range(L):
                print(batch.pools[i].dtype, batch.pools[i].shape)
            print('\nStack lengths')
            for i in range(L):
                print(batch.lengths[i].dtype, batch.lengths[i].shape)
            print('\nFeatures')
            print(batch.features.dtype, batch.features.shape)
            print('\nLabels')
            print(batch.labels.dtype, batch.labels.shape)
            print('\nAugment Scales')
            print(batch.scales.dtype, batch.scales.shape)
            print('\nAugment Rotations')
            print(batch.rots.dtype, batch.rots.shape)
            print('\nModel indices')
            print(batch.model_inds.dtype, batch.model_inds.shape)

            print('\nAre input tensors pinned')
            print(batch.neighbors[0].is_pinned())
            print(batch.neighbors[-1].is_pinned())
            print(batch.points[0].is_pinned())
            print(batch.points[-1].is_pinned())
            print(batch.labels.is_pinned())
            print(batch.scales.is_pinned())
            print(batch.rots.is_pinned())
            print(batch.model_inds.is_pinned())

            show_input_batch(batch)

        print('*******************************************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)


def debug_batch_and_neighbors_calib(dataset, loader):
    """Timing of generator function"""

    t = [time.time()]
    last_display = time.time()
    mean_dt = np.zeros(2)

    for epoch in range(10):

        for batch_i, input_list in enumerate(loader):
            # print(batch_i, tuple(points.shape),  tuple(normals.shape), labels, indices, in_sizes)

            # New time
            t = t[-1:]
            t += [time.time()]

            # Pause simulating computations
            time.sleep(0.01)
            t += [time.time()]

            # Average timing
            mean_dt = 0.9 * mean_dt + 0.1 * (np.array(t[1:]) - np.array(t[:-1]))

            # Console display (only one per second)
            if (t[-1] - last_display) > 1.0:
                last_display = t[-1]
                message = 'Step {:08d} -> Average timings (ms/batch) {:8.2f} {:8.2f} '
                print(message.format(batch_i,
                                     1000 * mean_dt[0],
                                     1000 * mean_dt[1]))

        print('************* Epoch ended *************')

    _, counts = np.unique(dataset.input_labels, return_counts=True)
    print(counts)
