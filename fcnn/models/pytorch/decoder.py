from functools import partial

from torch import nn

from .myronenko import MyronenkoLayer, MyronenkoResidualBlock
from . import resnet


class BasicDecoder(nn.Module):
    def __init__(self, in_planes, layers, block=resnet.BasicBlock, plane_dilation=2, upsampling_mode="trilinear",
                 upsampling_scale=2):
        super(BasicDecoder, self).__init__()
        self.layers = nn.ModuleList()
        self.conv1s = nn.ModuleList()
        self.upsampling_mode = upsampling_mode
        self.upsampling_scale = upsampling_scale
        layer_planes = in_planes
        for n_blocks in layers:
            self.conv1s.append(resnet.conv1x1x1(in_planes=layer_planes,
                                                out_planes=int(layer_planes/plane_dilation)))
            layer = nn.ModuleList()
            layer_planes = int(layer_planes/plane_dilation)
            for i_block in range(n_blocks):
                layer.append(block(layer_planes, layer_planes))
            self.layers.append(layer)

    def forward(self, x):
        for conv1, layer in zip(self.conv1s, self.layers):
            x = conv1(x)
            x = nn.functional.interpolate(x, scale_factor=self.upsampling_scale, mode=self.upsampling_mode)
            for block in layer:
                x = block(x)
        return x


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
            self.pre_upsampling_blocks.append(resnet.conv1x1x1(in_features, out_features, stride=1))
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