from torch import nn as nn
import numpy as np

from fcnn.models.pytorch.resnet import BasicBlock as ResNetBasicBlock
from fcnn.models.pytorch.resnet import conv3x3x3, conv1x1x1
from fcnn.models.pytorch.variational import VariationalBlock


class MyronenkoBlock(ResNetBasicBlock):
    def __init__(self, *args, norm_groups=8, **kwargs):
        self.norm_groups = norm_groups
        norm_layer = nn.GroupNorm
        if "norm_layer" in kwargs:
            raise RuntimeWarning(
                "Ignoring specified normalization layer, {}, and setting it to GroupNorm".format(norm_layer))
        kwargs["norm_layer"] = norm_layer
        kwargs["downsample"] = None
        super(MyronenkoBlock, self).__init__(*args, **kwargs)

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += identity

        return out

    def create_norm_layer(self, *args, **kwargs):
        return self.norm_layer(self.norm_groups, *args, **kwargs)


class MyronenkoLayer(nn.Module):
    def __init__(self, n_blocks, block, *args, **kwargs):
        super(MyronenkoLayer, self).__init__()
        self.block = block
        self.n_blocks = n_blocks
        self.blocks = list()
        for i in range(n_blocks):
            self.blocks.append(block(*args, **kwargs))
        self.layer = nn.Sequential(*self.blocks)

    def forward(self, x):
        return self.layer(x)


class MyronenkoVariationalLayer(nn.Module):
    def __init__(self, in_features, input_shape, reduced_features=16, latent_features=128, conv_block=MyronenkoBlock,
                 conv_stride=2, n_dims=3, upsampling_mode="trilinear", align_corners_upsampling=False):
        super(MyronenkoVariationalLayer, self).__init__()
        self.in_conv = conv_block(in_planes=in_features, planes=reduced_features, stride=conv_stride)
        self.reduced_shape = (reduced_features, *input_shape[-n_dims:])
        in_size = np.prod(self.reduced_shape)
        self.var_block = VariationalBlock(in_size=in_size, out_size=in_size, n_features=latent_features)
        self.relu = nn.ReLU()
        self.out_conv = conv1x1x1(in_planes=reduced_features, out_planes=in_features, stride=1)
        self.upsample = nn.Upsample(scale_factor=conv_stride, mode=upsampling_mode,
                                    align_corners=align_corners_upsampling)

    def forward(self, x):
        _x = x
        _x = self.in_conv(_x).flatten()
        _x, mu, logvar = self.var_block(_x)
        _x = self.relu(_x).reshape(self.reduced_shape)
        _x = self.out_conv(_x)
        _x = self.upsample(_x)
        return _x, mu, logvar


class MyronenkoEncoder(nn.Module):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoBlock, feature_dilation=2,
                 downsampling_stride=2):
        super(MyronenkoEncoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 2, 2, 4]
        self.layers = list()
        self.downsampling_convolutions = list()
        in_width = base_width
        for n_blocks in layer_blocks:
            out_width = base_width * feature_dilation
            self.layers.append(layer(n_blocks, block, in_planes=in_width, planes=out_width))
            if n_blocks != layer_blocks[-1]:
                self.downsampling_convolutions.append(conv3x3x3(out_width, out_width, stride=downsampling_stride))
            in_width = out_width

    def forward(self, x):
        _x = x
        for layer, downsampling in zip(self.layers[:-1], self.downsampling_convolutions):
            _x = layer(_x)
            _x = downsampling(_x)
        _x = self.layers[-1](_x)
        return _x


class MyronenkoDecoder(nn.Module):
    def __init__(self, base_width=32, layer_blocks=None, layer=MyronenkoLayer, block=MyronenkoBlock, upsampling_scale=2,
                 feature_reduction_scale=2, upsampling_mode="trilinear", align_corners=False):
        super(MyronenkoDecoder, self).__init__()
        if layer_blocks is None:
            layer_blocks = [1, 1, 1]
        self.layers = list()
        self.pre_upsampling_blocks = list()
        self.upsampling_blocks = list()
        for i, n_blocks in enumerate(layer_blocks):
            depth = len(layer_blocks) - (i + 1)
            out_features = base_width * (feature_reduction_scale ** depth)
            in_features = out_features * feature_reduction_scale
            self.pre_upsampling_blocks.append(conv1x1x1(in_features, out_features, stride=1))
            self.upsampling_blocks.append(nn.Upsample(scale_factor=upsampling_scale, mode=upsampling_mode,
                                                      align_corners=align_corners))
            self.layers.append(layer(n_blocks=n_blocks, block=block, in_planes=out_features, planes=out_features))

    def forward(self, x):
        _x = x
        for pre, up, lay in zip(self.pre_upsampling_blocks, self.upsampling_blocks, self.layers):
            _x = pre(_x)
            _x = up(_x)
            _x = lay(_x)
        return _x





