import torch
from mvpnet.ops.group_points import group_points
import mvpnet.models.unet_resnet34 as net2d
from mvpnet.models.mvpnet_3d import FeatureAggregation

#-------------------------------------
# Adapt the path
#-------------------------------------
PRETRAINED_2D_MODEL_PATH = "/home/dchangyu/MV-KPConv/outputs_use/scannet/unet_resnet34/model_080000.pth"
PRETRAINED_3D_MODEL_PATH = "/home/dchangyu/MV-KPConv/outputs_use/scannet/mvpnet_3d_unet_resnet34_pn2ssg/model_040000.pth"
CACHE_DIR = "/home/dchangyu/MV-KPConv/ScanNet/cache_rgbd"
IMAGE_DIR = "/home/dchangyu/MV-KPConv/ScanNet/scans_resize_160x120"
OUTPUT_DIR = "/home/dchangyu/MV-KPConv/"


def get_2d3dfeature(data_i, network_2d):
    data = data_i
    points = data['points']  # tensor (3,np)
    # colors = data.get('feature', None)
    images = data['images']  # (nv, 3, h, w) tensor

    b = 1
    nv = 30
    h = 120
    w = 160

    images = torch.from_numpy(images).unsqueeze(0)  # (1, nv, 3, h, w) ndarray -> tensor
    # collapse first 2 dimensions together
    images = images.reshape([-1] + list(images.shape[2:]))

    # load freezed 2D network

    net_2d = network_2d

    # net_2d.cuda()  #train 2d network on cpu to save cuda memory
    preds_2d = net_2d({'image': images})
    feature_2d = preds_2d['feature']  # (b * nv, c, h, w)

    # unproject features
    knn_indices = torch.from_numpy(data['knn_indices']).unsqueeze(0)  # (b, np, k)
    feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
    feature_2d = feature_2d.reshape(b, -1, nv * h * w)
    feature_2d = group_points(feature_2d.cuda(), knn_indices.cuda())  # (b, c, np, k)

    # unproject depth maps #(unproject point cloud direct from frames selected)
    with torch.no_grad():
        knn_indices = torch.from_numpy(data['knn_indices']).unsqueeze(0)
        image_xyz = torch.from_numpy(data['image_xyz']).unsqueeze(0)  # (b, nv, h, w, 3)
        image_xyz = image_xyz.permute(0, 4, 1, 2, 3).reshape(b, 3, nv * h * w)
        image_xyz = group_points(image_xyz.cuda(), knn_indices.cuda())  # (b, 3, np, k)

    # 2D-3D aggregation
    with torch.no_grad():
        points = torch.from_numpy(points).unsqueeze(0)  # (b,3,np)
        feat_aggreg = FeatureAggregation(64)

    #--------------------------------------------------#
    # load weights of FeatureAggregation from pretrained model
    # --------------------------------------------------#
        load_checkpoint = torch.load(PRETRAINED_3D_MODEL_PATH)
        load_model_parameter = load_checkpoint['model'].keys()
        feat_aggreg_weights = {}
        for key in load_model_parameter:
            for i in key.split('.'):
                if i == 'feat_aggreg':
                    keys = '.'.join(key.split('.')[1:])
                    feat_aggreg_weights[keys] = load_checkpoint['model'][key]

        feat_aggreg.load_state_dict(feat_aggreg_weights)
        feat_aggreg.eval()

        feat_aggreg.cuda()
        feature_2d3d = feat_aggreg(image_xyz.cuda(), points.cuda(), feature_2d.cuda())  # (b,64,np)

    cpu_feature_2d3d = feature_2d3d.squeeze().cpu() #(64,np)
    feature_2d3d_numpy = cpu_feature_2d3d.detach().numpy().T #(np,64)
    d = {'feature_2d3d' : feature_2d3d_numpy}

    return d

def get_2dfeature_imagexyz(data_i, network_2d):
    data = data_i
    points = data['points']  # tensor (3,np)
    # colors = data.get('feature', None)
    images = data['images']  # (nv, 3, h, w) tensor

    b = 1
    nv = 30
    h = 120
    w = 160
    k=3

    images = torch.from_numpy(images).unsqueeze(0)  # (1, nv, 3, h, w) ndarray -> tensor
    # collapse first 2 dimensions together
    images = images.reshape([-1] + list(images.shape[2:]))

    # load freezed 2D network

    net_2d = network_2d

    # net_2d.cuda()  #train 2d network on cpu to save cuda memory
    preds_2d = net_2d({'image': images})
    feature_2d = preds_2d['feature']  # (b * nv, c, h, w)

    # unproject features
    knn_indices = torch.from_numpy(data['knn_indices']).unsqueeze(0)  # (b, np, k)  #因为knn indices里面有np,所以resample点本质是限制knnindecis的数量，从而fix unproject的数量
    feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
    feature_2d = feature_2d.reshape(b, -1, nv * h * w)
    feature_2d = group_points(feature_2d.cuda(), knn_indices.cuda())  # (b, c, np, k) c=64
    feature_2d_cpu = feature_2d.squeeze().cpu()
    feature_2d_cpu = feature_2d_cpu.permute(1, 2 ,0).reshape(-1, 64*k) #(np,k*64)
    feature_2d_numpy = feature_2d_cpu.numpy()

    # unproject depth maps #(unproject point cloud direct from frames selected)
    with torch.no_grad():
        knn_indices = torch.from_numpy(data['knn_indices']).unsqueeze(0)
        image_xyz = torch.from_numpy(data['image_xyz']).unsqueeze(0)  # (b, nv, h, w, 3)
        image_xyz = image_xyz.permute(0, 4, 1, 2, 3).reshape(b, 3, nv * h * w)
        image_xyz = group_points(image_xyz.cuda(), knn_indices.cuda())  # (b, 3, np, k)
        image_xyz_cpu = image_xyz.squeeze().cpu()
        image_xyz_cpu = image_xyz_cpu.permute(1, 2, 0).reshape(-1, 3*k) #(np, k*3)
        image_xyz_numpy = image_xyz_cpu.numpy()


    d = {'feature_2d' : feature_2d_numpy, 'image_xyz' : image_xyz_numpy}

    return d

def get_2dfeature(data_i, network_2d):
    data = data_i
    points = data['points']  #  (3,np)
    # colors = data.get('feature', None)
    images = data['images']  # (nv, 3, h, w) tensor

    b = 1
    nv = 30
    h = 120
    w = 160
    k=3

    images = torch.from_numpy(images).unsqueeze(0)  # (1, nv, 3, h, w) ndarray -> tensor
    # collapse first 2 dimensions together
    images = images.reshape([-1] + list(images.shape[2:]))

    # load freezed 2D network

    net_2d = network_2d

    # net_2d.cuda()  #train 2d network on cpu to save cuda memory
    preds_2d = net_2d({'image': images})
    feature_2d = preds_2d['feature']  # (b * nv, c, h, w)

    # unproject features

    feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
    feature_2d = feature_2d.reshape(b, -1, nv * h * w) #####(b,64,576000)


    d = {'feature_2d' : feature_2d}

    return d



if __name__ == '__main__':
    import mvpnet.data.scannet_2d3d as scannet
    import os.path as osp
    import pickle

    cache_dir = osp.join(CACHE_DIR)
    # cache_dir = osp.join('/home/dchangyu/MV-KPConv/ScanNet/cache_rgbd_3scene')
    image_dir = osp.join(IMAGE_DIR)

    split = 'val'
    # label=0-20
    if split == 'train':
        dataset = scannet.ScanNet2D3DWhole(cache_dir=cache_dir,
                                           image_dir=image_dir,
                                           split=split,
                                           num_rgbd_frames=30,
                                           color_jitter=(0.4, 0.4, 0.4),
                                           flip=0.5,
                                           # z_rot=(-180, 180),
                                           to_tensor=True
                                           )
    else:
        dataset = scannet.ScanNet2D3DWhole(cache_dir=cache_dir,
                                           image_dir=image_dir,
                                           split=split,
                                           num_rgbd_frames=30,
                                           # color_jitter=(0.4, 0.4, 0.4),
                                           # flip=0.5,
                                           # z_rot=(-180, 180),
                                           to_tensor=True
                                           )

    #load 2d network
    net_2d = net2d.UNetResNet34(20, p=0.5, pretrained=True)

    checkpoint = torch.load(PRETRAINED_2D_MODEL_PATH,
                            map_location=torch.device("cpu"))
    net_2d.load_state_dict(checkpoint['model'])


    # build freezer
    for name, params in net_2d.named_parameters():
        print(name)
        params.requires_grad = False
    for name, m in net_2d._modules.items():
        print(name)
        m.train(False)

    data_is = []
    feature_list = []
    for index, data_i in enumerate(dataset):
        print('processing_scene_{}'.format(index))
        dic = get_2d3dfeature(data_i, net_2d)
        # dic = get_2dfeature_imagexyz(data_i,net_2d)
        print('save the label for scene_{}'.format(index))
        dic['seg_label'] = data_i['seg_label']
        print('save the points for scene_{}'.format(index))
        dic['points'] = data_i['points'].T
        print('save scan_id of this scene')
        dic['scan_id'] = data_i['scan_id']
        feature_list.append(dic)

    print('save the data')

    output_dir = OUTPUT_DIR
    output_path = osp.join(output_dir, 'scannet_2d3d_feature_{}.pkl'.format(split))
    # output_path = osp.join(output_dir, 'scannet2d3d_Whole_{}.pkl'.format(split))

    print('Save to {}'.format(osp.abspath(output_path)))
    with open(output_path, 'wb') as f:
        pickle.dump(feature_list, f, protocol=pickle.HIGHEST_PROTOCOL)