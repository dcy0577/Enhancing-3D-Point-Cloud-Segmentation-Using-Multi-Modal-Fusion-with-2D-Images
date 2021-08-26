#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling ScanNet dataset.
#      Implements a Dataset, a Sampler, and a collate_fn
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import numpy as np
import pickle
import torch
import math
from multiprocessing import Lock
import os.path as osp
from mvpnet.data.scannet_2d3d import ScanNet2D3DWhole
# OS functions
from os import listdir
from os.path import exists, join, isdir

# Dataset parent class
from datasets.common import PointCloudDataset
from torch.utils.data import Sampler, get_worker_info
from utils.mayavi_visu import *

from datasets.common import grid_subsampling
from utils.config import bcolors
import mvpnet.models.unet_resnet34 as net2d
from mvpnet.get_whole_scene_feature2d3d_augmented_pc import  get_2dfeature_imagexyz

from mvpnet.models.unet_resnet34 import UNetResNet34
from mvpnet.ops.group_points import group_points

# ----------------------------------------------------------------------------------------------------------------------
#
#           Dataset class definition
#       \******************************/


class ScanNetDataset(PointCloudDataset):
    """Class to handle Scannet dataset."""

    def __init__(self, config, set='training', use_potentials=False, load_data=True):
        """
        This dataset is small enough to be stored in-memory, so load all point clouds here
        """
        PointCloudDataset.__init__(self, 'ScanNet')

        ############
        # Parameters
        ############

        # Dict from labels to names
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

        # List of classes ignored during training (can be empty)
        #self.ignored_labels = np.array([])
        self.ignored_labels = np.array([-100])

        # Dataset folder
        self.path = config.dataset_path

        # Type of task conducted on this dataset
        self.dataset_task = 'cloud_segmentation'

        # Update number of class and data task in configuration
        config.num_classes = self.num_classes - len(self.ignored_labels)
        config.dataset_task = self.dataset_task

        # Parameters from config
        self.config = config

        # Training or test set
        self.set = set

        # Using potential or random epoch generation
        self.use_potentials = use_potentials

        #self.all_splits = [0]
        #self.validation_split = 0

        # Number of models used per epoch
        if self.set == 'training':
            self.epoch_n = config.epoch_steps * config.batch_num
        elif self.set in ['validation', 'test', 'ERF']:
            self.epoch_n = config.validation_size * config.batch_num
        else:
            raise ValueError('Unknown set for ScanNet data: ', self.set)

        # Stop data is not needed
        if not load_data:
            return


        ################
        # Load ply files
        ################


        with open('/home/dchangyu/MV-KPConv/scannet2d3d_Whole_val.pkl', 'rb') as f:
            data_val = pickle.load(f)
        with open('/home/dchangyu/MV-KPConv/scannet2d3d_Whole_train.pkl', 'rb') as f:
            data_train = pickle.load(f)

        if self.set == 'training':
            # self.files = ScanNet2D3DWhole(cache_dir=cache_dir,
            #                            image_dir=image_dir,
            #                            split= 'train',
            #                            # nb_pts=8192,
            #                            num_rgbd_frames=30,
            #                            # color_jitter=(0.4, 0.4, 0.4),
            #                            # flip=0.5,
            #                            # z_rot=(-180, 180),
            #                            to_tensor=True
            #                            )
            self.files = data_train

        elif self.set in ['validation', 'test', 'ERF']:
            # self.files = ScanNet2D3DWhole(cache_dir=cache_dir,
            #                            image_dir=image_dir,
            #                            split= 'val',
            #                            # nb_pts=8192,
            #                            num_rgbd_frames=30,
            #                            # color_jitter=(0.4, 0.4, 0.4),
            #                            # flip=0.5,
            #                            # z_rot=(-180, 180),
            #                            to_tensor=True
            #                            )

            self.files = data_val

        self.cloud_names = [data_i['scan_id'] for index, data_i in enumerate(self.files)]


        if 0 < self.config.first_subsampling_dl <= 0.01:
            raise ValueError('subsampling_parameter too low (should be over 1 cm')

        # Initiate containers
        self.input_trees = []
        self.input_colors = []
        #add 2 more extra containers
        self.input_image_xyz = []
        # self.input_knn_indices = []
        self.input_feature_2d = []
        #----------------------------
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
            self.potentials = None
            self.min_potentials = None
            self.argmin_potentials = None
            ##add if, else hier
            if self.set == 'training':
                N = config.epoch_steps * config.batch_num
            else:
                N = config.validation_size * config.batch_num #240
            self.epoch_inds = torch.from_numpy(np.zeros((2, N), dtype=np.int64))
            self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
            self.epoch_i.share_memory_()
            self.epoch_inds.share_memory_()

        self.worker_lock = Lock()

        # For ERF visualization, we want only one cloud per batch and no randomness
        if self.set == 'ERF':
            self.batch_limit = torch.tensor([1], dtype=torch.float32)
            self.batch_limit.share_memory_()
            np.random.seed(42)

        #######################################################################
        #2d network
        #####################################
        self.net_2d = UNetResNet34(20, p=0.5, pretrained=True)
        checkpoint = torch.load('/home/dchangyu/MV-KPConv/outputs/scannet/unet_resnet34/model_079000.pth',
                                map_location=torch.device("cpu"))
        self.net_2d.load_state_dict(checkpoint['model'])
        self.net_2d.eval()
        ################################################################################
        return

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
            return self.random_item(batch_i)

    def potential_item(self, batch_i, debug_workers=False):

        t = [time.time()]

        # Initiate concatanation lists
        p_list = []
        # f_list = []
        #NEW------------------
        p_f_list = []
        image_xyz_list = []
        feature_2d_list = []
        # knn_list = []
        #-----------------
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

                # Get potential points from tree structure
                pot_points = np.array(self.pot_trees[cloud_ind].data, copy=False)

                # Center point of input region
                center_point = pot_points[point_ind, :].reshape(1, -1)

                # Add a small noise to center point
                if self.set != 'ERF':
                    center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)

                # Indices of points in input region
                pot_inds, dists = self.pot_trees[cloud_ind].query_radius(center_point,
                                                                         r=self.config.in_radius,
                                                                         return_distance=True)

                d2s = np.square(dists[0])
                pot_inds = pot_inds[0]

                # Update potentials (Tukey weights)
                if self.set != 'ERF':
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
                                                                  r=self.config.in_radius)[0] #(np,)

            t += [time.time()]

            # Number collected
            n = input_inds.shape[0]

            # Collect labels and colors
            # ??????????????????????????
            input_points = (points[input_inds] - center_point).astype(np.float32) #(np,3)
            # input_colors = self.input_colors[cloud_ind][input_inds]

            # NEW: collect imagexyz and feature_2d
            input_image_xyz = self.input_image_xyz[cloud_ind]  # (b, 3, np, k) ndarray
            input_image_xyz = input_image_xyz[:,:,input_inds,:] # (b, 3, sub_np, k)
            input_feature_2d = self.input_feature_2d[cloud_ind]  # (b, c, np, k) ndarray
            input_feature_2d = input_feature_2d[:,:,input_inds,:] # (b, c, sub_np, k)




            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds].astype(np.int64)
                ######### 不转index！！ 原始网络input是index，而不是真实值！
                # input_labels = np.array([self.label_to_idx[l] for l in input_labels])


            t += [time.time()]


            # 把feat_aggre 所需的points以feature形式输入
            # input_points_feat_aggre = input_points ###input points 是 distance to center points???
            input_points_feat_aggre = points[input_inds].astype(np.float32)


            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            # if np.random.rand() > self.config.augment_color:
            #     input_colors *= 0

            # Get original height as additional feature
            # input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)

            # input_features = input_colors
            #这里把64作为additional feature, (?,64), input_colors 维度？




            t += [time.time()]

            # Stack batch
            p_list += [input_points]
            #--------------------
            p_f_list += [input_points_feat_aggre]
            # f_list += [feature_2d]
            image_xyz_list += [input_image_xyz]
            feature_2d_list += [input_feature_2d]
            # knn_list += [input_knn]
            #---------------------------
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

        stacked_points = np.concatenate(p_list, axis=0)
        # features = np.concatenate(f_list, axis=0)
        feat_aggre_points = np.concatenate(p_f_list, axis=0)

        image_xyz = np.concatenate(image_xyz_list, axis=2) #除了axis位置以外，其他维度应该相同
        feature_2d = np.concatenate(feature_2d_list, axis=2)
        # knn_indices = np.concatenate(knn_list,axis=0)
        #-------------------------------------------------
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        # stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        # #这里没用feature,但还是摆了个1
        # if self.config.in_features_dim == 1:
        #     pass
        # #这里用了color(3)
        # elif self.config.in_features_dim == 4:
        #     stacked_features = np.hstack((stacked_features, features[:, :3]))
        # #这里用了color(3) + height(1)
        # elif self.config.in_features_dim == 5:
        #     stacked_features = np.hstack((stacked_features, features))
        # #### 64 + 1 = 65
        # elif self.config.in_features_dim == 65:
        #     stacked_features = np.hstack((stacked_features, features))
        # #### 64 + 1 +1= 66
        # elif self.config.in_features_dim == 66:
        #     stacked_features = np.hstack((stacked_features, features))
        # else:
        #     raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        t += [time.time()]

        # Get the whole input list
        input_list = self.segmentation_inputs_new(stacked_points,
                                              image_xyz,
                                              feature_2d,
                                              feat_aggre_points,
                                              labels,
                                              stack_lengths)

        t += [time.time()]

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

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

    def random_item(self, batch_i):

        # Initiate concatanation lists
        p_list = []
        f_list = []
        l_list = []
        i_list = []
        pi_list = []
        ci_list = []
        s_list = []
        R_list = []
        batch_n = 0

        while True:

            with self.worker_lock:

                # Get potential minimum
                # ##break 如果超出
                if self.epoch_i == torch.from_numpy(np.array(self.epoch_n, )):  # 240
                    break
                ####self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))
                cloud_ind = int(self.epoch_inds[0, self.epoch_i]) #self.epoch_inds = torch.from_numpy(np.zeros((2, N), dtype=np.int64))
                point_ind = int(self.epoch_inds[1, self.epoch_i]) #self.epoch_i = torch.from_numpy(np.zeros((1,), dtype=np.int64))

                # Update epoch indice
                self.epoch_i += 1




            # Get points from tree structure
            points = np.array(self.input_trees[cloud_ind].data, copy=False)

            # Center point of input region
            center_point = points[point_ind, :].reshape(1, -1)

            # Add a small noise to center point
            if self.set != 'ERF':
                center_point += np.random.normal(scale=self.config.in_radius / 10, size=center_point.shape)

            # Indices of points in input region
            input_inds = self.input_trees[cloud_ind].query_radius(center_point,
                                                                  r=self.config.in_radius)[0]

            # Number collected
            n = input_inds.shape[0]

            # Collect labels and colors
            #input colors = feature_2d3d
            input_points = (points[input_inds] - center_point).astype(np.float32)
            input_colors = self.input_colors[cloud_ind][input_inds]
            if self.set in ['test', 'ERF']:
                input_labels = np.zeros(input_points.shape[0])
            else:
                input_labels = self.input_labels[cloud_ind][input_inds]
                input_labels = np.array([self.label_to_idx[l] for l in input_labels])

            # Data augmentation
            input_points, scale, R = self.augmentation_transform(input_points)

            # Color augmentation
            # if np.random.rand() > self.config.augment_color:
            #     input_colors *= 0

            # Get original height as additional feature
            #input_features = np.hstack((input_colors, input_points[:, 2:] + center_point[:, 2:])).astype(np.float32)
            input_features = input_colors

            # Stack batch
            p_list += [input_points]
            f_list += [input_features]
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

        stacked_points = np.concatenate(p_list, axis=0)
        features = np.concatenate(f_list, axis=0)
        labels = np.concatenate(l_list, axis=0)
        point_inds = np.array(i_list, dtype=np.int32)
        cloud_inds = np.array(ci_list, dtype=np.int32)
        input_inds = np.concatenate(pi_list, axis=0)
        stack_lengths = np.array([pp.shape[0] for pp in p_list], dtype=np.int32)
        scales = np.array(s_list, dtype=np.float32)
        rots = np.stack(R_list, axis=0)

        # Input features
        stacked_features = np.ones_like(stacked_points[:, :1], dtype=np.float32)
        if self.config.in_features_dim == 1:
            pass
        elif self.config.in_features_dim == 4:
            stacked_features = np.hstack((stacked_features, features[:, :3]))
        elif self.config.in_features_dim == 5:
            stacked_features = np.hstack((stacked_features, features))
        elif self.config.in_features_dim == 65:
            stacked_features = np.hstack((stacked_features, features))
        # height as additional feature: 64+height+1
        # elif self.config.in_features_dim == 66:
        #     stacked_features = np.hstack((stacked_features, features))
        # height, color as additional feature: 64+height+RGB+1
        # elif self.config.in_features_dim == 69:
        #     stacked_features = np.hstack((stacked_features, features))
        else:
            raise ValueError('Only accepted input dimensions are 1, 4 and 7 (without and with XYZ)')

        #######################
        # Create network inputs
        #######################
        #
        #   Points, neighbors, pooling indices for each layers
        #

        # Get the whole input list
        input_list = self.segmentation_inputs(stacked_points,
                                              stacked_features,
                                              labels,
                                              stack_lengths)

        # Add scale and rotation for testing
        input_list += [scales, rots, cloud_inds, point_inds, input_inds]

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

        dataset = self.files

        for index, data_i in enumerate(dataset):

            # Restart timer
            t0 = time.time()




            # Get cloud name
            # cloud_name = data_i['scan_id'] #改进！！！！！！！！！！！！
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
                #sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
                feature_2d = data_load['feature_2d']
                image_xyz = data_load['image_xyz']
                # sub_knn_indices = data_load['sub_knn_indices'] #(np,k)
                sub_labels = data_load['sub_labels']

                # Read pkl with search tree
                with open(KDTree_file, 'rb') as f:
                    search_tree = pickle.load(f)

            else:
                print('\nPreparing KDTree for cloud {:s}, subsampled at {:.3f}'.format(cloud_name, dl))

                # 2dnet work start to get imagexyz and feature2d
                net_2d = net2d.UNetResNet34(20, p=0.5, pretrained=True)

                checkpoint = torch.load('/home/dchangyu/mvpnet/outputs_use/scannet/unet_resnet34/model_080000.pth',
                                        map_location=torch.device("cpu"))
                net_2d.load_state_dict(checkpoint['model'])
                # build freezer
                for name, params in net_2d.named_parameters():
                    # print(name)
                    params.requires_grad = False
                for name, m in net_2d._modules.items():
                    # print(name)
                    m.train(False)

                # imagexyz_feature2d_dic = get_2dfeature_imagexyz(data_i, net_2d)
                # Read ply file

                points = data_i['points'].T  #(n,3) #
                #colors = np.vstack((data['red'], data['green'], data['blue'])).T #(n,3) #grid_sample可以考虑feature(n,d)? #
                #64feature 附加在此？
                knn_indices = data_i['knn_indices'] #（np,k） k=3
                knn_indices = np.squeeze(knn_indices).astype(np.float32) #(np,k)
                images = data_i['images'] # ( num_views, 3, h, w) nv=30
                image_xyz = data_i['image_xyz'] #( nv, h, w, 3)

                # image_xyz = imagexyz_feature2d_dic['image_xyz'] #(np, k*3)
                # feature_2d = imagexyz_feature2d_dic['feature_2d'] #(np,k*64)

                labels= data_i['seg_label'].astype(np.int32) #int64

                # map labels to its ids: -100->0, 0->1, ...., 19 -> 20


                # Subsample cloud
                sub_points, sub_knn_indices, sub_labels = grid_subsampling(points,
                                                                      features=knn_indices,
                                                                      labels=labels,
                                                                      sampleDl=dl)

                ###################################################################################################

                images = np.expand_dims(images, axis=0)  # (batch_size, num_views, 3, h, w)
                # b, nv, _, h, w = images.shape
                b = images.shape[0]
                nv = images.shape[1]
                _ = images.shape[2]
                h = images.shape[3]
                w = images.shape[4]

                # collapse first 2 dimensions together
                images = images.reshape([-1] + list(images.shape[2:]))

                images = torch.from_numpy(images)

                print('start 2d net work processing #################################')

                # 2D network

                preds_2d = net_2d({'image': images})
                feature_2d = preds_2d['feature']  # (b * nv, c, h, w)
                # feature_2d = feature_2d.cuda()

                # unproject features
                knn_indices = sub_knn_indices  # (np, k) float32
                knn_indices = np.expand_dims(knn_indices, axis=0) # (1, np, k) int
                knn_indices = torch.from_numpy(knn_indices).long()
                feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
                feature_2d = feature_2d.reshape(b, -1, nv * h * w)  #####(b,64,576000)
                feature_2d = group_points(feature_2d.cuda(), knn_indices.cuda())  # (b, c, np, k)
                feature_2d = feature_2d.cpu().detach().numpy()

                # unproject depth maps #(unproject point cloud direct from frames selected)
                with torch.no_grad():
                    image_xyz = image_xyz  # (nv, h, w, 3)
                    image_xyz = np.expand_dims(image_xyz, axis=0)  # (b, nv, h, w, 3)
                    image_xyz = torch.from_numpy(image_xyz)
                    image_xyz = image_xyz.permute(0, 4, 1, 2, 3).reshape(b, 3, nv * h * w)  ####(b,3,576000) 30*120*160=576000
                    image_xyz = group_points(image_xyz.cuda(), knn_indices.cuda())  # (b, 3, np, k)
                    image_xyz = image_xyz.cpu().detach().numpy()

                ##################################################################################################

                # Rescale float color and squeeze label
                #sub_colors = sub_colors / 255

                sub_labels = np.squeeze(sub_labels)

                # Get chosen neighborhoods
                search_tree = KDTree(sub_points, leaf_size=10)
                #search_tree = nnfln.KDTree(n_neighbors=1, metric='L2', leaf_size=10)
                #search_tree.fit(sub_points)

                # Save KDTree
                with open(KDTree_file, 'wb') as f:
                    pickle.dump(search_tree, f)

                # Save pkl save points feature and labels
                sub_data = {'sub_points': sub_points, 'image_xyz' : image_xyz, 'feature_2d' : feature_2d, 'sub_labels' : sub_labels}
                with open(sub_pkl_file, 'wb') as f:
                    pickle.dump(sub_data, f)



            # Fill data containers
            self.input_trees += [search_tree]
            # self.input_colors += [sub_feature_2d3d]
            # extra 2 fill data containers
            self.input_feature_2d += [feature_2d]
            self.input_image_xyz += [image_xyz]
            # self.input_knn_indices += [sub_knn_indices]
            #————————————————————————————————————————————————————
            self.input_labels += [sub_labels]

            size1 = feature_2d.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size1 * 1e-6, time.time() - t0))
            size2 = image_xyz.shape[0] * 4 * 7
            print('{:.1f} MB loaded in {:.1f}s'.format(size2 * 1e-6, time.time() - t0))
            # size3 = images.shape[0] * 4 * 7
            # print('{:.1f} MB loaded in {:.1f}s'.format(size3 * 1e-6, time.time() - t0))

        ############################
        # Coarse potential locations
        ############################

        # Only necessary for validation and test sets
        if self.use_potentials:
            print('\nPreparing potentials')

            # Restart timer
            t0 = time.time()

            pot_dl = self.config.in_radius / 10
            cloud_ind = 0

            for index, data_i in enumerate(dataset):

                # Get cloud name
                # cloud_name = data_i['scan_id'] #改进！！太慢了等于每次跑一编dataset
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

        # Get number of clouds
        self.num_clouds = len(self.input_trees)



        # Only necessary for validation and test sets
        if self.set in ['validation', 'test']:

            print('\nPreparing reprojection indices for testing')

            # Get validation/test reprojection indices
            #这里for是对的
            for index, data_i in enumerate(dataset):

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
                    #data = read_ply(file_path)
                    points = data_i['points'].T
                    labels = data_i['seg_label'].astype(np.int32)

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
        #file_path传进来的是pkl文件列表中的各个字典
        return file_path['points'].T


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
        like Semantic3D, before potential have been updated over the dataset, there are cahnces that all the dense area
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

    def calibration(self, dataloader, untouched_ratio=0.9, verbose=False, force_redo=False):
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
        # L = (len(input_list) - 7) // 5
        #解释在https://github.com/HuguesTHOMAS/KPConv-PyTorch/issues/69
        L = (len(input_list) - 9) // 5

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
        # self.features = torch.from_numpy(input_list[ind])
        self.feat_aggre_points = torch.from_numpy(input_list[ind])
        ind += 1
        #----------------------------------------------------------------
        self.image_xyz = torch.from_numpy(input_list[ind])
        ind += 1
        self.feature_2d = torch.from_numpy(input_list[ind])

        #--------------------------------------------------------------------------
        ind += 1
        self.labels = torch.from_numpy(input_list[ind])
        ind += 1
        ##imagexyz, feature 2d
        self.scales = torch.from_numpy(input_list[ind])
        ind += 1
        self.rots = torch.from_numpy(input_list[ind])
        ind += 1
        self.cloud_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.center_inds = torch.from_numpy(input_list[ind])
        ind += 1
        self.input_inds = torch.from_numpy(input_list[ind])

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
        # self.features = self.features.pin_memory()
        self.feat_aggre_points = self.feat_aggre_points.pin_memory()

        self.image_xyz = self.image_xyz.pin_memory()
        self.feature_2d = self.feature_2d.pin_memory()

        #-------------------------------------------------
        self.labels = self.labels.pin_memory()
        self.scales = self.scales.pin_memory()
        self.rots = self.rots.pin_memory()
        self.cloud_inds = self.cloud_inds.pin_memory()
        self.center_inds = self.center_inds.pin_memory()
        self.input_inds = self.input_inds.pin_memory()

        return self

    def to(self, device):

        self.points = [in_tensor.to(device) for in_tensor in self.points]
        self.neighbors = [in_tensor.to(device) for in_tensor in self.neighbors]
        self.pools = [in_tensor.to(device) for in_tensor in self.pools]
        self.upsamples = [in_tensor.to(device) for in_tensor in self.upsamples]
        self.lengths = [in_tensor.to(device) for in_tensor in self.lengths]
        # self.features = self.features.to(device)
        self.feat_aggre_points = self.feat_aggre_points.to(device)

        self.image_xyz = self.image_xyz.to(device)
        self.feature_2d = self.feature_2d.to(device)

        #=============================
        self.labels = self.labels.to(device)
        self.scales = self.scales.to(device)
        self.rots = self.rots.to(device)
        self.cloud_inds = self.cloud_inds.to(device)
        self.center_inds = self.center_inds.to(device)
        self.input_inds = self.input_inds.to(device)

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
