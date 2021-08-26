import torch
from torch import nn
import numpy as np

from common.nn import SharedMLP

from mvpnet.ops.group_points import group_points
import mvpnet.models.unet_resnet34 as net2d
from functools import partial


class FeatureAggregation(nn.Module):
    """Feature Aggregation inspired by ContFuse"""

    def __init__(self,
                 in_channels,
                 mlp_channels=(64, 64, 64),
                 reduction='sum',
                 use_relation=True,
                 ):
        super(FeatureAggregation, self).__init__()

        self.in_channels = in_channels
        self.use_relation = use_relation

        if mlp_channels:
            self.out_channels = mlp_channels[-1]
            self.mlp = SharedMLP(in_channels + (4 if use_relation else 0), mlp_channels, ndim=2, bn=True) #use conv2d, kernel=1X1
        else:
            self.out_channels = in_channels
            self.mlp = None

        if reduction == 'sum':
            self.reduction = torch.sum
        elif reduction == 'max':
            self.reduction = lambda x, dim: torch.max(x, dim)[0]

        self.reset_parameters()

    def forward(self, src_xyz, tgt_xyz, feature):
        """

        Args:
            src_xyz (torch.Tensor): (batch_size, 3, num_points, k)
            tgt_xyz (torch.Tensor): (batch_size, 3, num_points)
            feature (torch.Tensor): (batch_size, in_channels, num_points, k)

        Returns:
            torch.Tensor: (batch_size, out_channels, num_points)

        """
        if self.mlp is not None:
            if self.use_relation:
                diff_xyz = src_xyz - tgt_xyz.unsqueeze(-1)  # (b, 3, np, k) #-1自动识别增加在哪个维度
                distance = torch.sum(diff_xyz ** 2, dim=1, keepdim=True)  # (b, 1, np, k)
                relation_feature = torch.cat([diff_xyz, distance], dim=1) # (b, 4, np, k)
                x = torch.cat([feature, relation_feature], 1) # (b, inchannels+4, np, k)
            else:
                x = feature # (b, inchannels, np, k)
            x = self.mlp(x) #sharedMLP
            x = self.reduction(x, 3) #sum, dim=3求和之后这个dim的元素个数为１，所以要被去掉，如果要保留这个维度，则应当keepdim=True,这步为了去除k
        else:
            x = self.reduction(feature, 3)#为了去除k,达到return时候的数据维度
        return x

    def reset_parameters(self):
        from common.nn.init import xavier_uniform
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)


class MVPNet3D(nn.Module):
    def __init__(self,
                 net_2d,
                 net_2d_ckpt_path,
                 net_3d,
                 **feat_aggr_kwargs,
                 ):
        super(MVPNet3D, self).__init__()
        self.net_2d = net_2d
        if net_2d_ckpt_path: #2D net check point
            checkpoint = torch.load(net_2d_ckpt_path, map_location=torch.device("cpu"))
            self.net_2d.load_state_dict(checkpoint['model'])
            import logging
            logger = logging.getLogger(__name__)
            logger.info("2D network load weights from {}.".format(net_2d_ckpt_path))
        self.feat_aggreg = FeatureAggregation(**feat_aggr_kwargs) #??
        self.net_3d = net_3d

    def forward(self, data_batch):
        # (batch_size, num_views, 3, h, w)
        images = data_batch['images']
        b, nv, _, h, w = images.size()
        # collapse first 2 dimensions together
        images = images.reshape([-1] + list(images.shape[2:]))

        # 2D network
        preds_2d = self.net_2d({'image': images})
        feature_2d = preds_2d['feature']  # (b * nv, c, h, w) requires_grad=false

        # unproject features
        knn_indices = data_batch['knn_indices']  # (b, np, k)
        feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
        feature_2d = feature_2d.reshape(b, -1, nv * h * w) #(b,64,576000) 30*120*160=576000
        feature_2d = group_points(feature_2d, knn_indices)  # (b, c, np, k) requires_grad=false

        # unproject depth maps
        with torch.no_grad():
            image_xyz = data_batch['image_xyz']  # (b, nv, h, w, 3)
            image_xyz = image_xyz.permute(0, 4, 1, 2, 3).reshape(b, 3, nv * h * w)  #(b,3,576000) 30*120*160=576000
            image_xyz = group_points(image_xyz, knn_indices)  # (b, 3, np, k)

        # 2D-3D aggregation
        points = data_batch['points'] #(b,3,np)
        feature_2d3d = self.feat_aggreg(image_xyz, points, feature_2d) #(b,64,np)

        # 3D network
        preds_3d = self.net_3d({'points': points, 'feature': feature_2d3d})
        preds = preds_3d
        return preds

    def get_loss(self, cfg):
        from mvpnet.models.loss import SegLoss
        if cfg.TRAIN.LABEL_WEIGHTS_PATH:
            weights = np.loadtxt(cfg.TRAIN.LABEL_WEIGHTS_PATH, dtype=np.float32)
            weights = torch.from_numpy(weights).cuda()
        else:
            weights = None
        return SegLoss(weight=weights)

    def get_metric(self, cfg):
        from mvpnet.models.metric import SegAccuracy, SegIoU
        metric_fn = lambda: [SegAccuracy(), SegIoU(self.net_3d.num_classes)]
        return metric_fn(), metric_fn()

def test():
    import mvpnet.data.scannet_2d3d as scannet
    import os.path as osp

    cache_dir = osp.join('/home/dchangyu/MV-KPConv/ScanNet/cache_rgbd')
    image_dir = osp.join('/home/dchangyu/MV-KPConv/ScanNet/scans_resize_160x120')

    np.random.seed(0)
    dataset = scannet.ScanNet2D3DWhole(cache_dir=cache_dir,
                                    image_dir=image_dir,
                                    split='val',
                                    #nb_pts=8192,
                                    num_rgbd_frames=20,
                                    # color_jitter=(0.5, 0.5, 0.5),
                                    # flip=0.5,
                                    # z_rot=(-180, 180),
                                    to_tensor=True
                                    )
    print(dataset)
    for i in range(len(dataset)):
        data = dataset[i]
        points = data['points'] #tensor (3,np)
        # colors = data.get('feature', None)
        for k, v in data.items():
            if isinstance(v, np.ndarray):
                print(k, v.shape, v.dtype)
            else:
                print('below not ndarray')
                print(k, v)

        images = data['images'] # (nv, 3, h, w) tensor
        image_xyz = data['image_xyz']

        print('store pc data'+str(i))

        from mvpnet.ops.group_points import group_points
        import torch
        b = 1
        nv = 20
        h = 120
        w = 160

        images = torch.from_numpy(images).unsqueeze(0) # (1, nv, 3, h, w) ndarray -> tensor
        # collapse first 2 dimensions together
        images = images.reshape([-1] + list(images.shape[2:]))

        # 2D network
        import mvpnet.models.unet_resnet34 as net2d
        net_2d = net2d.UNetResNet34(20, p=0.5 ,pretrained=True)

        checkpoint = torch.load('/home/dchangyu/MV-KPConv/outputs/scannet/unet_resnet34/model_080000.pth', map_location=torch.device("cpu"))
        net_2d.load_state_dict(checkpoint['model'])
        #net_2d.cuda()  #train 2d network on cpu to save cuda memory

        from common.nn.freezer import Freezer
        # build freezer
        freezer = Freezer(net_2d, ("module:net_2d", "net_2d"))
        freezer.freeze(verbose=True)  # sanity check

        preds_2d = net_2d({'image': images})
        feature_2d = preds_2d['feature']  # (b * nv, c, h, w)

        print('feature_2d from net_2d', feature_2d.shape, feature_2d.dtype, feature_2d.device)

        # unproject features
        knn_indices = torch.from_numpy(data['knn_indices']).unsqueeze(0)  # (b, np, k)  #因为knn indices里面有np,所以resample点本质是限制knnindecis的数量，从而fix unproject的数量
        feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
        feature_2d = feature_2d.reshape(b, -1, nv * h * w)
        feature_2d = group_points(feature_2d.cuda(), knn_indices.cuda())  # (b, c, np, k)

        print('feature_2d', feature_2d.shape, feature_2d.dtype, feature_2d.device)

        # unproject depth maps #(unproject point cloud direct from frames selected)
        with torch.no_grad():
            knn_indices = torch.from_numpy(data['knn_indices']).unsqueeze(0)
            image_xyz = torch.from_numpy(data['image_xyz']).unsqueeze(0)  # (b, nv, h, w, 3)
            image_xyz = image_xyz.permute(0,4, 1, 2, 3).reshape(b, 3, nv * h * w)
            image_xyz = group_points(image_xyz.cuda(), knn_indices.cuda())  # (b, 3, np, k)

        print('image_xyz_group_points',image_xyz.shape, image_xyz.dtype, image_xyz.device)

        # 2D-3D aggregation
        points = torch.from_numpy(data['points']).unsqueeze(0)  # (b,3,np)
        feat_aggreg = FeatureAggregation(64)
        feat_aggreg.cuda()
        feature_2d3d = feat_aggreg(image_xyz.cuda(), points.cuda(), feature_2d.cuda())  # (b,64,np)

        print('feature_2d3d', feature_2d3d.shape, feature_2d3d.dtype, feature_2d3d.device)
        print(feature_2d3d)


        torch.cuda.empty_cache()
        print('cuda empty after one loop')

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
    knn_indices = torch.from_numpy(data['knn_indices']).unsqueeze(0)  # (b, np, k)  #因为knn indices里面有np,所以resample点本质是限制knnindecis的数量，从而fix unproject的数量
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
    # load weights from pretrained model
    # --------------------------------------------------#
        load_checkpoint = torch.load(
            '/home/dchangyu/MV-KPConv/outputs_use/scannet/mvpnet_3d_unet_resnet34_pn2ssg/model_040000.pth')
        load_model_parameter = load_checkpoint['model'].keys()
        feat_aggreg_weights = {}
        for key in load_model_parameter:
            for i in key.split('.'):
                if i == 'feat_aggreg':
                    keys = '.'.join(key.split('.')[1:])
                    feat_aggreg_weights[keys] = load_checkpoint['model'][key]

        feat_aggreg.load_state_dict(feat_aggreg_weights)
        feat_aggreg.eval()
# -----------------------------------------------------------#

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

if __name__ == '__main__':
    import mvpnet.data.scannet_2d3d as scannet
    import os.path as osp
    import pickle

    cache_dir = osp.join('/home/dchangyu/MV-KPConv/ScanNet/cache_rgbd')
    # cache_dir = osp.join('/home/dchangyu/MV-KPConv/ScanNet/cache_rgbd_3scene')
    image_dir = osp.join('/home/dchangyu/MV-KPConv/ScanNet/scans_resize_160x120')

    split = 'val'
    dataset = scannet.ScanNet2D3DWhole(cache_dir=cache_dir,
                                       image_dir=image_dir,
                                       split= split,
                                       # nb_pts=8192,
                                       num_rgbd_frames=30,
                                       # color_jitter=(0.4, 0.4, 0.4),
                                       # flip=0.5,
                                       # z_rot=(-180, 180),
                                       to_tensor=True
                                       )

    #load 2d network
    net_2d = net2d.UNetResNet34(20, p=0.5, pretrained=True)

    checkpoint = torch.load('/home/dchangyu/MV-KPConv/outputs_use/scannet/unet_resnet34/model_080000.pth',
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
    for index in range(len(dataset)):
        data_i = dataset[index]
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

    output_dir = '/home/dchangyu/MV-KPConv/'
    output_path = osp.join(output_dir, 'scannet_2d3d_feature_{}.pkl'.format(split))

    print('Save to {}'.format(osp.abspath(output_path)))
    with open(output_path, 'wb') as f:
        pickle.dump(feature_list, f, protocol=pickle.HIGHEST_PROTOCOL)

    #test()

