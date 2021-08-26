# @inproceedings{jaritz2019multi,
# 	title={Multi-view PointNet for 3D Scene Understanding},
# 	author={Jaritz, Maximilian and Gu, Jiayuan and Su, Hao},
# 	booktitle={ICCV Workshop 2019},
# 	year={2019}
# }

import os
import os.path as osp
import pickle
import time
import argparse
from functools import partial
import glob

import numpy as np
import natsort
from plyfile import PlyData
from PIL import Image
import open3d as o3d

DATA_DIR = '/home/dchangyu/MV-KPConv/ScanNet'
SCAN_DIR = 'scans_resize_160x120'
# Here the validation data are also stored in scans, if use test data, they should be stored in scans_test
SCAN_TEST_DIR = 'scans'
META_DIR = '/home/dchangyu/MV-KPConv/mvpnet/data/meta_files'

SEG_CLASS_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
# INST_CLASS_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
_DEBUG = False
# _DEBUG = True

# exclude some frames with problematic data (e.g. depth frames with zeros everywhere or unreadable labels)
exclude_frames = {
    'scene0243_00': ['1175', '1176', '1177', '1178', '1179', '1180', '1181', '1182', '1183', '1184'],
    'scene0538_00': ['1925', '1928', '1929', '1931', '1932', '1933'],
    'scene0639_00': ['442', '443', '444'],
    'scene0299_01': ['1512'],
}


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def get_rgbd_paths(scan_dir, scan_id):
    return {
        'color': osp.join(scan_dir, 'color', '{}.jpg'),
        'depth': osp.join(scan_dir, 'depth', '{}.png'),
        'pose': osp.join(scan_dir, 'pose', '{}.txt'),
        'intrinsics_depth': osp.join(scan_dir, 'intrinsic', 'intrinsic_depth.txt'),
    }


def unproject(k, depth_map, mask=None):
    if mask is None:
        # only consider points where we have a depth value
        mask = depth_map > 0
    # create xy coordinates from image position
    v, u = np.indices(depth_map.shape)
    v = v[mask]
    u = u[mask]
    depth = depth_map[mask].ravel()
    uv1_points = np.stack([u, v, np.ones_like(u)], axis=1)
    points_3d_xyz = (np.linalg.inv(k[:3, :3]).dot(uv1_points.T) * depth).T
    return points_3d_xyz


def compute_rgbd_knn(cloud_name, frame_ids, cam_matrix, paths, subsampeled_scene_pts,
                     num_base_pts=6000, resize=(80, 60), debug = False): #more base points more accurate
    # choose m base points
    base_point_ind = np.random.choice(len(subsampeled_scene_pts), num_base_pts, replace=False)
    base_pts = subsampeled_scene_pts[base_point_ind]

    # initialize output
    overlaps = np.zeros([len(base_point_ind), len(frame_ids)], dtype=bool)

    # build kd tree for base points
    base_pts_pc = o3d.geometry.PointCloud()
    base_pts_pc.points = o3d.utility.Vector3dVector(base_pts)
    pcd_tree = o3d.geometry.KDTreeFlann(base_pts_pc)

    if resize:
        # Note that we may use 160x120 depth maps; however, camera matrix here is irrelevant to that.
        # adjust intrinsic matrix
        depth_map_size = (640, 480)
        cam_matrix = cam_matrix.copy()  # avoid overwriting
        cam_matrix[0] /= depth_map_size[0] / resize[0]
        cam_matrix[1] /= depth_map_size[1] / resize[1]

    last_time = time.time()
    for i, frame_id in enumerate(frame_ids):
        if (i + 1 % 1000) == 0:
            now = time.time()
            print('[{}/{}] time: {:.2f}'.format(i + 1, len(frame_ids), now - last_time))
            last_time = now

        # load pose
        pose = np.loadtxt(paths['pose'].format(frame_id))
        if np.any(np.isinf(pose)):
            print('Skipping frame {}, because pose is not valid.'.format(frame_id))
            continue

        # load depth map
        depth = Image.open(paths['depth'].format(frame_id))
        if resize:
            depth = depth.resize(resize, Image.NEAREST)
        depth = np.asarray(depth, dtype=np.float32) / 1000.

        # un-project point cloud from depth map
        unproj_pts = unproject(cam_matrix, depth)

        # apply pose to unprojected points
        unproj_pts = pose[:3, :3].dot(unproj_pts[:, :3].T).T + pose[:3, 3]

        # for each point of unprojected point cloud find nearest neighbor (only one!) in whole scene point cloud(whole scene base points)
        for j in range(len(unproj_pts)):
            # find a neighbor that is at most 1cm away
            found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(unproj_pts[j, :3], 0.1, 1)
            if found:
                overlaps[idx_point, i] = True

        # visualize
        # debug = True
        if debug:
            from mvpnet.utils.o3d_util import draw_point_cloud
            pts_vis = draw_point_cloud(subsampeled_scene_pts)
            base_pts_vis = draw_point_cloud(base_pts, colors=[1., 0., 0.])
            overlap_base_pts_vis = draw_point_cloud(base_pts[overlaps[:, i]], colors=[0., 1., 0.])
            unproj_pts_vis = draw_point_cloud(unproj_pts, colors=[0., 0., 1.])
            # o3d.visualization.draw_geometries([overlap_base_pts_vis, unproj_pts_vis, base_pts_vis, pts_vis])
            # o3d.visualization.draw_geometries([base_pts_vis, unproj_pts_vis, overlap_base_pts_vis])
            path = '/home/dchangyu/mvpnet/'
            o3d.io.write_point_cloud(path + cloud_name + 'base_points' + '.ply', base_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'overlap_base_pts_vis' + '.ply', overlap_base_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'unproj_pts_vis' + '.ply', unproj_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'subsampeled_scene_pts' + '.ply',  pts_vis)

    return base_point_ind, overlaps


# ----------------------------------------------------------------------------- #
# Worker function
# ----------------------------------------------------------------------------- #

def test():
    image_dir = '/home/dchangyu/mvpnet/ScanNet/scans_resize_160x120'
    scan_id = 'scene0000_02'
    scan_dir = osp.join(image_dir, scan_id)
    cache_dir = '/home/dchangyu/mvpnet/ScanNet/cache_rgbd'
    split = 'train'
    # split = 'val'
    with open(osp.join(cache_dir, 'scannetv2_{}.pkl'.format(split)), 'rb') as f:
        cache_data = pickle.load(f)

    exclude_frames = {
        'scene0243_00': ['1175', '1176', '1177', '1178', '1179', '1180', '1181', '1182', '1183', '1184'],
        'scene0538_00': ['1925', '1928', '1929', '1931', '1932', '1933'],
        'scene0639_00': ['442', '443', '444'],
        'scene0299_01': ['1512'],
    }

    paths = {'color': osp.join(scan_dir, 'color', '{}.jpg'),
        'depth': osp.join(scan_dir, 'depth', '{}.png'),
        'pose': osp.join(scan_dir, 'pose', '{}.txt'),
        'intrinsics_depth': osp.join(scan_dir, 'intrinsic', 'intrinsic_depth.txt'),
    }

    glob_path = osp.join(scan_dir, 'color', '*')
    cam_matrix = np.loadtxt(paths['intrinsics_depth'], dtype=np.float32)
    color_paths = natsort.natsorted(glob.glob(glob_path))
    exclude_ids = exclude_frames.get(scan_id, [])
    frame_ids = [osp.splitext(osp.basename(x))[0] for x in color_paths]
    frame_ids = [x for x in frame_ids if x not in exclude_ids]
    if not frame_ids:
        print('WARNING: No frames found, check glob path {}'.format(glob_path))

    base_point_ind, pointwise_rgbd_overlap = compute_rgbd_knn(frame_ids, cam_matrix, paths, cache_data['points']) # cache_data return whole scence points

    print(base_point_ind)
    print(pointwise_rgbd_overlap)

if __name__ == '__main__':
    test()
