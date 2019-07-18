from functools import partial

from torch import nn as nn
import numpy as np

from fcnn.models.pytorch.resnet import conv3x3x3, conv1x1x1
from fcnn.models.pytorch.variational import VariationalBlock


class MyronenkoConvolutionBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8):
        super(MyronenkoConvolutionBlock, self).__init__()
        self.norm_groups = norm_groups
        if norm_layer is None:
            self.norm_layer = nn.GroupNorm
        else:
            self.norm_layer = norm_layer
        self.norm1 = self.create_norm_layer(in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv = conv3x3x3(in_planes, planes, stride)

    def forward(self, x):
        out = self.norm1(x)
        out = self.relu(out)
        out = self.conv(out)
        return out

    def create_norm_layer(self, planes):
        if planes < self.norm_groups:
            return self.norm_layer(planes, planes)
        else:
            return self.norm_layer(self.norm_groups, planes)


class MyronenkoResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, norm_layer=None, norm_groups=8):
        super(MyronenkoResidualBlock, self).__init__()
        self.conv1 = MyronenkoConvolutionBlock(in_planes=in_planes, planes=planes, stride=stride, norm_layer=norm_layer,
                                               norm_groups=norm_groups)
        self.conv2 = MyronenkoConvolutionBlock(in_planes=planes, planes=planes, stride=stride, norm_layer=norm_layer,
                                               norm_groups=norm_groups)
        if in_planes != planes:
            self.sample = conv1x1x1(in_planes, planes)
        else:
            self.sample = None

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.conv2(out)

        if self.sample is not None:
            identity = self.sample(x)

        out += identity

        return out


class MyronenkoLayer(nn.Module):
    def __init__(self, n_blocks, block, in_planes, planes, *args, dropout=None, **kwargs):
        super(MyronenkoLayer, self).__init__()
        self.block = block
        self.n_blocks = n_blocks
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            self.blocks.append(block(in_planes, planes, *args, **kwargs))
            in_planes = planes
        if dropout is not None:
            self.dropout = nn.Dropout3d(dropout, inplace=True)
        else:
            self.dropout = None

    def forward(self, x):
        _x = x
        for i, block in enumerate(self.blocks):
            _x = block(_x)
            if i == 0 and self.dropout is not None:
                _x = self.dropout(_x)
        return _x


class MyronenkoVariationalLayer(nn.Module):
    def __init__(self, in_features, input_shape, reduced_features=16, latent_features=128,
                 conv_block=MyronenkoConvolutionBlock, conv_stride=2, upsampling_mode="trilinear",
                 align_corners_upsampling=False):
        super(MyronenkoVariationalLayer, self).__init__()
        self.in_conv = conv_block(in_planes=in_features, planes=reduced_features, stride=conv_stride)
        self.reduced_shape = tuple(np.asarray((reduced_features, *np.divide(input_shape, 2)), dtype=np.int))
        self.in_size = np.prod(self.reduced_shape, dtype=np.int)
        self.var_block = VariationalBlock(in_size=self.in_size, out_size=self.in_size, n_features=latent_features)
        self.relu = nn.ReLU(inplace=True)
        self.out_conv = conv1x1x1(in_planes=reduced_features, out_planes=in_features, stride=1)
        self.upsample = partial(nn.functional.interpolate, scale_factor=conv_stride, mode=upsampling_mode,
                                align_corners=align_corners_upsampling)

    def forward(self, x):
        _x = x
        _x = self.in_conv(_x).flatten(start_dim=1)
        _x, mu, logvar = self.var_block(_x)
        _x = self.relu(_x).view(-1, *self.reduced_shape)
        _x = self.out_conv(_x)
        _x = self.upsample(_x)
        return _x, mu, logvar


class MyronenkoEncoder(nn.Module):
    def __init__(self, n_features, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 feature_dilation=2, downsampling_stride=2, dropout=0.2):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        self.layers = nn.ModuleList()
        self.downsampling_convolutions = nn.ModuleList()
        out_width = base_width
        in_width = n_features
        for i, n_blocks in enumerate(layer_blocks):
            if dropout and i == 0:
                layer_dropout = dropout
            else:
                layer_dropout = None
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=in_width, planes=out_width,
                                     dropout=layer_dropout))
            if i != len(layer_blocks) - 1:
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride))
            in_width = out_width
            out_width = out_width * feature_dilation

    def forward(self, x):
        _x = x
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            _x = layer(_x)
            _x = downsampling(_x)
        _x = self.layers[-1](_x)
        return _x


class MyronenkoDecoder(nn.Module):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoResidualBlock,
                 upsampling_scale=2, feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False):
        super(MyronenkoDecoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 1, 1]
        self.layers = nn.ModuleList()
        self.pre_upsampling_blocks = nn.ModuleList()
        self.upsampling_blocks = list()
        for i, n_blocks in enumerate(layer_blocks):
            depth = len(layer_blocks) - (i + 1)
            out_features = base_width * (feature_reduction_scale ** depth)
            in_features = out_features * feature_reduction_scale
            self.pre_upsampling_blocks.append(conv1x1x1(in_features, out_features, stride=1))
            self.upsampling_blocks.append(partial(nn.functional.interpolate, scale_factor=upsampling_scale,
                                                  mode=upsampling_mode, align_corners=align_corners))
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=out_features, planes=out_features))

    def forward(self, x):
        _x = x
        for pre, up, lay in zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers):
            _x = pre(_x)
            _x = up(_x)
            _x = lay(_x)
        return _x





