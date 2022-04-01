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

from colmap.python.read_write_model import read_images_binary, read_cameras_binary
from colmap.python.read_write_dense import read_array


Colmap_root = '/home/dchangyu/MV-KPConv/colmap/'

# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def get_data_paths(colmap_root):
    return {
        'color': osp.join(colmap_root, 'color', '{}.jpg'),
        'depth': osp.join(colmap_root, 'depth', '{}.png'),
        'pose': osp.join(colmap_root, 'pose', '{}.txt'),
        'intrinsics_depth': osp.join(colmap_root, 'intrinsic', 'intrinsic_depth.txt'),
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


#----------------------------------------------------------------------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------#

def compute_rgbd_knn_colmap(cloud_name, cameras_dic,  image_info_dic, paths, subsampeled_scene_pts,
                     num_base_pts=2000, resize = (380, 200), debug =False):
    # choose m base points
    base_point_ind = np.random.choice(len(subsampeled_scene_pts), num_base_pts, replace=False)
    base_pts = subsampeled_scene_pts[base_point_ind]

    # initialize output
    # frame_ids is key list, start from 15, end at 183, but not continuous
    frame_ids = [key for key in image_info_dic.keys()]
    max_key = max(frame_ids) # 183
    overlaps = np.zeros([len(base_point_ind), max_key+1], dtype=bool)

    # build kd tree for base points
    base_pts_pc = o3d.geometry.PointCloud()
    base_pts_pc.points = o3d.utility.Vector3dVector(base_pts)
    pcd_tree = o3d.geometry.KDTreeFlann(base_pts_pc)

    # "Camera", ["id", "model", "width", "height", "params"])
    # cameras_dic = read_cameras_binary(paths['camera_intrinsics'])
    # "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    # image_info_dic = read_images_binary(paths['pose'])

    for i, image in image_info_dic.items():

        # load pose #(rotation matrix and translation vector)
        rotation_matrix = image.qvec2rotmat()
        t = image.tvec

        # load intrinsic
        fx = cameras_dic[image.camera_id].params[0]
        fy = cameras_dic[image.camera_id].params[1]
        cx = cameras_dic[image.camera_id].params[2]
        cy = cameras_dic[image.camera_id].params[3]
        # build intrinsic matrix
        cam_matrix = np.zeros(shape=(3,3))
        cam_matrix[0] = [fx,0,cx]
        cam_matrix[1] = [0,fy,cy]
        cam_matrix[2] = [0,0,1]

        # load depth map (H,W,C)
        depth = read_array(paths['depth'].format(image.name))

        # # resize = (160, 120) # w,h
        # resize = (380, 200)  # w,h # we dont need to resize to 80 60 because the depth map from colmap is already very sparse
        # # resize = (80, 60)  # w,h  # for overlap compute we use 80*60 depth map to speed up

        if resize:
            # original size around w=1920, h= 1080 but not stable, differ from images to images
            # Note that we may use 1900x1000 depth maps; however, camera matrix here is irrelevant to that.
            # adjust intrinsic matrix
            depth_map_size = (depth.shape[1], depth.shape[0]) # (w,h)
            cam_matrix = cam_matrix.copy()  # avoid overwriting
            cam_matrix[0] /= depth_map_size[0] / resize[0]
            cam_matrix[1] /= depth_map_size[1] / resize[1]
        # numpy -> Image
        depth_im = Image.fromarray(depth) #(w,h)?
        if resize: #resize to (380,200)
            depth_im = depth_im.resize(resize, Image.NEAREST)
        # depth = np.asarray(depth_im, dtype=np.float32) / 1000.
        depth = np.asarray(depth_im, dtype=np.float32)

        # un-project point cloud from depth map
        unproj_pts = unproject(cam_matrix, depth)

        # apply pose to unprojected points
        unproj_pts = np.matmul(unproj_pts - t, rotation_matrix)

        # aligns unproject points with scan point cloud
        matrix = np.loadtxt(paths['translation_matrix_for_images'], dtype=np.float32)
        ones = np.ones(shape=unproj_pts.shape[0])
        unproj_pts = np.append(unproj_pts.T, [ones], axis=0)
        unproj_pts = np.matmul(unproj_pts.T, matrix.T)
        unproj_pts = np.delete(unproj_pts,3, axis=1)

        # rot_matrix = matrix[:3,:3]
        # t_matrix = matrix[:3,3]
        # unproj_pts = np.matmul(unproj_pts, rot_matrix) - t_matrix

        # extrinsic matrix
        # x = np.column_stack((rotation_matrix,t))
        # row = np.array([0,0,0,1])
        # x = np.append(x,[row],axis=0)
        #
        # ones = np.ones(shape=unproj_pts.shape[0])
        # unproj_pts = np.append(unproj_pts.T, [ones], axis=0)
        #
        # unproj_pts = np.matmul(unproj_pts.T, x.T)
        #
        # unproj_pts = np.delete(unproj_pts,3, axis=1)


        # for each point of unprojected point cloud find nearest neighbor (only one!) in whole scene point cloud(whole scene base points)
        for j in range(len(unproj_pts)):
            # find a neighbor that is at most 1cm away
            found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(unproj_pts[j, :3], 0.1, 1)
            if found:
                # i is key of image_dic
                overlaps[idx_point, i] = True

        # visualize
        if debug:
            from mvpnet.utils.o3d_util import draw_point_cloud
            pts_vis = draw_point_cloud(subsampeled_scene_pts)
            base_pts_vis = draw_point_cloud(base_pts, colors=[1., 0., 0.])
            overlap_base_pts_vis = draw_point_cloud(base_pts[overlaps[:, i]], colors=[0., 1., 0.])
            unproj_pts_vis = draw_point_cloud(unproj_pts, colors=[0., 0., 1.])
            # o3d.visualization.draw_geometries([overlap_base_pts_vis, unproj_pts_vis, base_pts_vis, pts_vis])
            # o3d.visualization.draw_geometries([base_pts_vis, unproj_pts_vis, overlap_base_pts_vis])
            path = '/home/dchangyu/MV-KPConv/'
            o3d.io.write_point_cloud(path + cloud_name + 'base_points' + '.ply', base_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'overlap_base_pts_vis' + '.ply', overlap_base_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'unproj_pts_vis' + '.ply', unproj_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'subsampeled_scene_pts' + '.ply', pts_vis)

    return base_point_ind, overlaps, frame_ids



# ----------------------------------------------------------------------------- #
# Worker function
# ----------------------------------------------------------------------------- #

def test():
    path_parameter = '/home/dchangyu/MV-KPConv/colmap/parameter'
    path_depth_maps = '/home/dchangyu/MV-KPConv/colmap/depth_maps'
    path_color_images = '/home/dchangyu/MV-KPConv/colmap/images'

    paths = {'color': osp.join(path_color_images, '{}'),
             # 'depth': osp.join(path_depth_maps, '{}.photometric.bin'),
             'depth': osp.join(path_depth_maps, '{}.geometric.bin'),
             'pose': osp.join(path_parameter, 'images.bin'),
             'camera_intrinsics': osp.join(path_parameter, 'cameras.bin'),
             }

    # "Camera", ["id", "model", "width", "height", "params"])
    cameras_dic = read_cameras_binary(paths['camera_intrinsics'])
    # "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
    image_info_dic = read_images_binary(paths['pose'])

    # frame_ids = [133, 134, 135]
    frame_ids = [15, 16, 17] # im_133, im_134, im_135

    # # choose m base points
    # num_base_pts = 4000
    # base_point_ind = np.random.choice(len(subsampeled_scene_pts), num_base_pts, replace=False)
    # base_pts = subsampeled_scene_pts[base_point_ind]

    # initialize output
    # overlaps = np.zeros([len(base_point_ind), len(frame_ids)], dtype=bool)

    # # build kd tree for base points
    # base_pts_pc = o3d.geometry.PointCloud()
    # base_pts_pc.points = o3d.utility.Vector3dVector(base_pts)
    # pcd_tree = o3d.geometry.KDTreeFlann(base_pts_pc)

    last_time = time.time()
    for key, image in image_info_dic.items():

        # load pose #(rotation matrix and translation vector)
        rotation_matrix = image.qvec2rotmat()
        t = image.tvec

        # load intrinsic
        fx = cameras_dic[image.camera_id].params[0]
        fy = cameras_dic[image.camera_id].params[1]
        cx = cameras_dic[image.camera_id].params[2]
        cy = cameras_dic[image.camera_id].params[3]
        # build intrinsic matrix
        cam_matrix = np.zeros(shape=(3,3))
        cam_matrix[0] = [fx,0,cx]
        cam_matrix[1] = [0,fy,cy]
        cam_matrix[2] = [0,0,1]

        # load depth map (H,W,C)
        depth = read_array(paths['depth'].format(image.name))

        # resize = (160, 120) # w,h
        resize = (380, 200)  # w,h # we dont need to resize to 80 60 because the depth map from colmap is already very sparse
        # resize = (80, 60)  # w,h  # for overlap compute we use 80*60 depth map to speed up

        if resize:
            # original size around w=1920, h= 1080 but not stable, differ from images to images
            # Note that we may use 1900x1000 depth maps; however, camera matrix here is irrelevant to that.
            # adjust intrinsic matrix
            depth_map_size = (depth.shape[1], depth.shape[0]) # (w,h)
            cam_matrix = cam_matrix.copy()  # avoid overwriting
            cam_matrix[0] /= depth_map_size[0] / resize[0]
            cam_matrix[1] /= depth_map_size[1] / resize[1]
        # numpy -> Image
        depth_im = Image.fromarray(depth) #(w,h)?
        if resize: #resize to (380,200)
            depth_im = depth_im.resize(resize, Image.NEAREST)
        # depth = np.asarray(depth_im, dtype=np.float32) / 1000.
        depth = np.asarray(depth_im, dtype=np.float32)

        # un-project point cloud from depth map
        unproj_pts = unproject(cam_matrix, depth)

        # apply pose to unprojected points
        unproj_pts = np.matmul(unproj_pts - t, rotation_matrix)

        # extrinsic matrix
        # x = np.column_stack((rotation_matrix,t))
        # row = np.array([0,0,0,1])
        # x = np.append(x,[row],axis=0)
        #
        # ones = np.ones(shape=unproj_pts.shape[0])
        # unproj_pts = np.append(unproj_pts.T, [ones], axis=0)
        #
        # unproj_pts = np.matmul(unproj_pts.T, x.T)
        #
        # unproj_pts = np.delete(unproj_pts,3, axis=1)

        # # for each point of unprojected point cloud find nearest neighbor (only one!) in whole scene point cloud(whole scene base points)
        # for j in range(len(unproj_pts)):
        #     # find a neighbor that is at most 1cm away
        #     found, idx_point, dist = pcd_tree.search_hybrid_vector_3d(unproj_pts[j, :3], 0.1, 1)
        #     if found:
        #         overlaps[idx_point, i] = True

        # visualize
        debug = True
        if debug:
            from mvpnet.utils.o3d_util import draw_point_cloud
            # pts_vis = draw_point_cloud(subsampeled_scene_pts)
            # base_pts_vis = draw_point_cloud(base_pts, colors=[1., 0., 0.])
            # overlap_base_pts_vis = draw_point_cloud(base_pts[overlaps[:, i]], colors=[0., 1., 0.])
            unproj_pts_vis = draw_point_cloud(unproj_pts, colors=[0., 0., 1.])
            # o3d.visualization.draw_geometries([overlap_base_pts_vis, unproj_pts_vis, base_pts_vis, pts_vis])
            # o3d.visualization.draw_geometries([base_pts_vis, unproj_pts_vis, overlap_base_pts_vis])
            path = '/home/dchangyu/MV-KPConv/'
            cloud_name = 'colmap_test_image'+ image.name
            # o3d.io.write_point_cloud(path + cloud_name + 'base_points' + '.ply', base_pts_vis)
            # o3d.io.write_point_cloud(path + cloud_name + 'overlap_base_pts_vis' + '.ply', overlap_base_pts_vis)
            o3d.io.write_point_cloud(path + cloud_name + 'unproj_pts_vis' + '.ply', unproj_pts_vis)
            # o3d.io.write_point_cloud(path + cloud_name + 'subsampeled_scene_pts' + '.ply', pts_vis)


if __name__ == '__main__':
    test()
