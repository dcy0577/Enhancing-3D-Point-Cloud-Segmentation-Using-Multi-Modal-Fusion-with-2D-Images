from torch import nn
import torch.nn.functional as F

from .conv import Conv1dBNReLU, Conv2dBNReLU
from .linear import LinearBNReLU


class MLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 bn=True):
        """Multi-layer perception with relu activation

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            bn (bool): whether to use batch normalization

        """
        super(MLP, self).__init__() #子类重写__init__()方法又需要调用父类的方法：使用super关键词

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]

        c_in = in_channels
        for ind, c_out in enumerate(mlp_channels): #ind为元组元素对应的index,c_out为元素的实际值
            self.append(LinearBNReLU(c_in, c_out, relu=True, bn=bn)) #因为for给mlp填加了多个LinearBNReLU层模块即fc+BN+RElu
            c_in = c_out

    def forward(self, x):
        for module in self:
            assert isinstance(module, LinearBNReLU) #assert断言，如果isinstance是false则触发错误报告中断程序，isinstance判断type
            x = module(x) #??
        return x


class SharedMLP(nn.ModuleList):
    def __init__(self,
                 in_channels,
                 mlp_channels,
                 ndim=1,
                 bn=True):
        """Multi-layer perception shared on resolution (1D or 2D)

        Args:
            in_channels (int): the number of channels of input tensor
            mlp_channels (tuple): the numbers of channels of fully connected layers
            ndim (int): the number of dimensions to share
            bn (bool): whether to use batch normalization

        """
        super(SharedMLP, self).__init__()

        self.in_channels = in_channels
        self.out_channels = mlp_channels[-1]
        self.ndim = ndim

        if ndim == 1:
            mlp_module = Conv1dBNReLU
        elif ndim == 2:
            mlp_module = Conv2dBNReLU
        else:
            raise ValueError('SharedMLP only supports ndim=(1, 2).')

        c_in = in_channels
        for ind, c_out in enumerate(mlp_channels):
            self.append(mlp_module(c_in, c_out, 1, relu=True, bn=bn)) #很多个conv1d/conv2d + bn+relu层, 1 是kernel size
            c_in = c_out

    def forward(self, x):
        for module in self:
            assert isinstance(module, (Conv1dBNReLU, Conv2dBNReLU))
            x = module(x)
        return x


class SharedMLPDO(SharedMLP):
    """Shared MLP with dropout"""

    def __init__(self, *args, p=0.5, **kwargs):
        super(SharedMLPDO, self).__init__(*args, **kwargs)
        self.p = p
        self.dropout_fn = F.dropout if self.ndim == 1 else F.dropout2d

    def forward(self, x):
        for module in self:
            assert isinstance(module, (Conv1dBNReLU, Conv2dBNReLU))
            x = module(x)
            # Note that inplace does not work.
            x = self.dropout_fn(x, p=self.p, training=self.training, inplace=False)
        return x

    def extra_repr(self):
        return 'p={}'.format(self.p)
