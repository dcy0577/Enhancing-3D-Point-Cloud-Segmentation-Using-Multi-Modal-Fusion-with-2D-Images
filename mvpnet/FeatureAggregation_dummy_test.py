import torch
from torch import nn


from common.nn import SharedMLP

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
            self.mlp = SharedMLP(in_channels + (4 if use_relation else 0), mlp_channels, ndim=2, bn=True)
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
            x = self.reduction(x, 3)
        else:
            x = self.reduction(feature, 3)
        return x

    def reset_parameters(self):
        from common.nn.init import xavier_uniform
        for m in self.modules():
            if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Linear)):
                xavier_uniform(m)

def test():
    # with torch.no_grad():
    src_xyz = torch.empty(1,3,5,3).fill_(32.)
    tgt_xyz = torch.empty(1,3,5).fill_(64.)
    feature = torch.empty(1,64,5,3).fill_(16.)

    feat_aggreg = FeatureAggregation(64)

    # for name, params in feat_aggreg.named_parameters():
    #     print(name)
    #     params.requires_grad = False
    # for name, m in feat_aggreg._modules.items():
    #     print(name)
    #     m.train(False)

    load_checkpoint = torch.load('/home/dchangyu/MV-KPConv/outputs_old/scannet/mvpnet_3d_unet_resnet34_pn2ssg/model_040000.pth')
    load_model_parameter = load_checkpoint['model'].keys()
    feat_aggreg_weights = {}
    for key in load_model_parameter:
        for i in key.split('.'):
            if i == 'feat_aggreg':
                keys = '.'.join(key.split('.')[1:])
                feat_aggreg_weights[keys] = load_checkpoint['model'][key]

    feat_aggreg.load_state_dict(feat_aggreg_weights)

    # for name, m in feat_aggreg._modules.items():
    #     print(name)
    #     m.train(False)
    feat_aggreg.eval()

    feat_aggreg.cuda()
    feature_2d3d = feat_aggreg(src_xyz.cuda(), tgt_xyz.cuda(), feature.cuda())  # (b,64,np)

    print('feature_2d3d', feature_2d3d.shape, feature_2d3d.dtype, feature_2d3d.device)
    print(feature_2d3d)

if __name__ == '__main__':
    test()