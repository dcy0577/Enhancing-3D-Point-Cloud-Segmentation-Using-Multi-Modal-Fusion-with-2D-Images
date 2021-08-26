#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Define network architectures
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 06/03/2020
#

from models.blocks import *
import numpy as np
from mvpnet.models.mvpnet_3d import FeatureAggregation
from mvpnet.models.unet_resnet34 import UNetResNet34
from mvpnet.ops.group_points import group_points


def p2p_fitting_regularizer(net):

    fitting_loss = 0
    repulsive_loss = 0

    for m in net.modules():

        if isinstance(m, KPConv) and m.deformable:

            ##############
            # Fitting loss
            ##############

            # Get the distance to closest input point and normalize to be independant from layers
            KP_min_d2 = m.min_d2 / (m.KP_extent ** 2)

            # Loss will be the square distance to closest input point. We use L1 because dist is already squared
            fitting_loss += net.l1(KP_min_d2, torch.zeros_like(KP_min_d2))

            ################
            # Repulsive loss
            ################

            # Normalized KP locations
            KP_locs = m.deformed_KP / m.KP_extent

            # Point should not be close to each other
            for i in range(net.K):
                other_KP = torch.cat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], dim=1).detach()
                distances = torch.sqrt(torch.sum((other_KP - KP_locs[:, i:i + 1, :]) ** 2, dim=2))
                rep_loss = torch.sum(torch.clamp_max(distances - net.repulse_extent, max=0.0) ** 2, dim=1)
                repulsive_loss += net.l1(rep_loss, torch.zeros_like(rep_loss)) / net.K

    return net.deform_fitting_power * (2 * fitting_loss + repulsive_loss)


class KPFCNN_featureAggre(nn.Module):
    """
    Class defining KPFCNN
    """

    def __init__(self, config, lbl_values, ign_lbls):
        super(KPFCNN_featureAggre, self).__init__()

        ############
        # Parameters
        ############

        # Current radius of convolution and feature dimension
        layer = 0
        r = config.first_subsampling_dl * config.conv_radius
        in_dim = config.in_features_dim
        out_dim = config.first_features_dim
        self.K = config.num_kernel_points
        self.C = len(lbl_values) - len(ign_lbls)

        #####################
        # List Encoder blocks
        #####################

        # Save all block operations in a list of modules
        self.encoder_blocks = nn.ModuleList()
        self.encoder_skip_dims = []
        self.encoder_skips = []

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture):

            # Check equivariance
            if ('equivariant' in block) and (not out_dim % 3 == 0):
                raise ValueError('Equivariant block but features dimension is not a factor of 3')

            # Detect change to next layer for skip connection
            if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
                self.encoder_skips.append(block_i)
                self.encoder_skip_dims.append(in_dim)

            # Detect upsampling block to stop
            if 'upsample' in block:
                break

            # Apply the good block function defining tf ops
            self.encoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            if 'simple' in block:
                in_dim = out_dim // 2
            else:
                in_dim = out_dim

            # Detect change to a subsampled layer
            if 'pool' in block or 'strided' in block:
                # Update radius and feature dimension for next layer
                layer += 1
                r *= 2
                out_dim *= 2

        #####################
        # List Decoder blocks
        #####################

        # Save all block operations in a list of modules
        self.decoder_blocks = nn.ModuleList()
        self.decoder_concats = []

        # Find first upsampling block
        start_i = 0
        for block_i, block in enumerate(config.architecture):
            if 'upsample' in block:
                start_i = block_i
                break

        # Loop over consecutive blocks
        for block_i, block in enumerate(config.architecture[start_i:]):

            # Add dimension of skip connection concat
            if block_i > 0 and 'upsample' in config.architecture[start_i + block_i - 1]:
                in_dim += self.encoder_skip_dims[layer]
                self.decoder_concats.append(block_i)

            # Apply the good block function defining tf ops
            self.decoder_blocks.append(block_decider(block,
                                                    r,
                                                    in_dim,
                                                    out_dim,
                                                    layer,
                                                    config))

            # Update dimension of input from output
            in_dim = out_dim

            # Detect change to a subsampled layer
            if 'upsample' in block:
                # Update radius and feature dimension for next layer
                layer -= 1
                r *= 0.5
                out_dim = out_dim // 2

        self.head_mlp = UnaryBlock(out_dim, config.first_features_dim, False, 0)
        self.head_softmax = UnaryBlock(config.first_features_dim, self.C, False, 0)

        ################
        # Network Losses
        ################

        # List of valid labels (those not ignored in loss)
        self.valid_labels = np.sort([c for c in lbl_values if c not in ign_lbls])

        # Choose segmentation loss
        if len(config.class_w) > 0:
            class_w = torch.from_numpy(np.array(config.class_w, dtype=np.float32))
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_w, ignore_index=-1)
        else:
            self.criterion = torch.nn.CrossEntropyLoss(ignore_index=-1)
        self.deform_fitting_mode = config.deform_fitting_mode
        self.deform_fitting_power = config.deform_fitting_power
        self.deform_lr_factor = config.deform_lr_factor
        self.repulse_extent = config.repulse_extent
        self.output_loss = 0
        self.reg_loss = 0
        self.l1 = nn.L1Loss()

        ################
        # Feature Aggregation
        ################
        self.feat_aggreg = FeatureAggregation(64)

        #load model
        # load_checkpoint = torch.load(
        #     '/home/dchangyu/mvpnet/outputs_use/scannet/mvpnet_3d_unet_resnet34_pn2ssg/model_040000.pth')
        # load_model_parameter = load_checkpoint['model'].keys()
        # feat_aggreg_weights = {}
        # for key in load_model_parameter:
        #     for i in key.split('.'):
        #         if i == 'feat_aggreg':
        #             keys = '.'.join(key.split('.')[1:])
        #             feat_aggreg_weights[keys] = load_checkpoint['model'][key]
        #
        # self.feat_aggreg.load_state_dict(feat_aggreg_weights)
        #
        # self.feat_aggreg.eval()

        ################
        # 2d network
        ################

        # self.net_2d = UNetResNet34(20, p=0.5, pretrained=True)
        # checkpoint = torch.load('/home/dchangyu/mvpnet/outputs_use/scannet/unet_resnet34/model_080000.pth',
        #                         map_location=torch.device("cpu"))
        # self.net_2d.load_state_dict(checkpoint['model'])
        # self.net_2d.eval()
        # self.net_2d.cpu()

        return

    def forward(self, batch, config):

        # images = batch.images # (num_views, 3, h, w) tensor
        # images = images.unsqueeze(0) # (batch_size, num_views, 3, h, w)
        # b, nv, _, h, w = images.size()
        # # collapse first 2 dimensions together
        # images = images.reshape([-1] + list(images.shape[2:]))
        #
        # # 2D network
        # self.net_2d.cpu()
        # preds_2d = self.net_2d({'image': images.cpu()})
        # feature_2d = preds_2d['feature']  # (b * nv, c, h, w)
        # feature_2d = feature_2d.cuda()
        #
        # # unproject features
        # knn_indices = batch.knn_indices  # (np, k) float32
        # knn_indices = knn_indices.unsqueeze(0).long() # (1, np, k) int
        # feature_2d = feature_2d.reshape(b, nv, -1, h, w).transpose(1, 2).contiguous()  # (b, c, nv, h, w)
        # feature_2d = feature_2d.reshape(b, -1, nv * h * w)  #####(b,64,576000)
        # feature_2d = group_points(feature_2d, knn_indices)  # (b, c, np, k)
        #
        # # unproject depth maps #(unproject point cloud direct from frames selected)
        # with torch.no_grad():
        #     image_xyz = batch.image_xyz  # (nv, h, w, 3)
        #     image_xyz = image_xyz.unsqueeze(0) # (b, nv, h, w, 3)
        #     image_xyz = image_xyz.permute(0, 4, 1, 2, 3).reshape(b, 3, nv * h * w)  ####(b,3,576000) 30*120*160=576000
        #     image_xyz = group_points(image_xyz, knn_indices)  # (b, 3, np, k)

        # # 2D-3D aggregation
        # feat_aggre_points = batch.feat_aggre_points  # (np,3)
        # feat_aggre_points = feat_aggre_points.unsqueeze(0).reshape(3, -1)  # (b,3,np)
        # feature_2d3d = self.feat_aggreg(batch.image_xyz, feat_aggre_points, batch.feature_2d)  # (b,64,np)
        # feature_2d3d = feature_2d3d.squeeze(0).T  # (np,64)


        #feature aggregation

        # with torch.no_grad():
        #     image_xyz = batch.image_xyz #(np, k*3)
        #     image_xyz = image_xyz.reshape(3, -1, 3).unsqueeze(0) # (b, 3, np, k=3)
        #
        # feature_2d = batch.feature_2d #(np, k*64)
        # feature_2d = feature_2d.reshape(64, -1, 3).unsqueeze(0) # (b, c=64, np, k)
        # points = batch.points[0] #first one in list -> for first layer, hast same np like feature
        # points = points.unsqueeze(0).reshape(3,-1) #(b,3,np)

        feat_aggre_points = batch.feat_aggre_points #(np,3)
        feat_aggre_points = feat_aggre_points.unsqueeze(0).reshape(1,3,-1) #(b,3,np)

        feature_2d3d = self.feat_aggreg(batch.image_xyz,feat_aggre_points,batch.feature_2d) #(b,64,np)
        feature_2d3d = feature_2d3d.squeeze(0).T #(np,64)

        #feature stack with 1?
        stacked_features = torch.ones_like(batch.feat_aggre_points[:, :1]) #(np,1)
        stacked_features = torch.cat((stacked_features, feature_2d3d),dim=1) #(np,65)


        # Get input features
        # x = batch.features.clone().detach()
        x = stacked_features.clone() #64+1
        # x = feature_2d3d.clone()

        # Loop over consecutive blocks
        skip_x = []
        for block_i, block_op in enumerate(self.encoder_blocks):
            if block_i in self.encoder_skips:
                skip_x.append(x)
            x = block_op(x, batch)

        for block_i, block_op in enumerate(self.decoder_blocks):
            if block_i in self.decoder_concats:
                x = torch.cat([x, skip_x.pop()], dim=1)
            x = block_op(x, batch)

        # Head of network
        x = self.head_mlp(x, batch)
        x = self.head_softmax(x, batch)

        return x

    def loss(self, outputs, labels):
        """
        Runs the loss on outputs of the model
        :param outputs: logits
        :param labels: labels
        :return: loss
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        if torch.equal(target, labels):
            print('ok')

        # Reshape to have a minibatch size of 1
        outputs = torch.transpose(outputs, 0, 1)
        outputs = outputs.unsqueeze(0)
        target = target.unsqueeze(0)

        # Cross entropy loss
        self.output_loss = self.criterion(outputs, target)

        # Regularization of deformable offsets
        if self.deform_fitting_mode == 'point2point':
            self.reg_loss = p2p_fitting_regularizer(self)
        elif self.deform_fitting_mode == 'point2plane':
            raise ValueError('point2plane fitting mode not implemented yet.')
        else:
            raise ValueError('Unknown fitting mode: ' + self.deform_fitting_mode)

        # Combined loss
        return self.output_loss + self.reg_loss

    def accuracy(self, outputs, labels):
        """
        Computes accuracy of the current batch
        :param outputs: logits predicted by the network
        :param labels: labels
        :return: accuracy value
        """

        # Set all ignored labels to -1 and correct the other label to be in [0, C-1] range
        target = - torch.ones_like(labels)
        for i, c in enumerate(self.valid_labels):
            target[labels == c] = i

        predicted = torch.argmax(outputs.data, dim=1)
        total = target.size(0)
        correct = (predicted == target).sum().item()

        return correct / total





















