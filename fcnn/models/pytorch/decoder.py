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


class Decoder1D(nn.Module):
    def __init__(self, input_features, output_features, layer_blocks, layer_channels, block=resnet.BasicBlock1D,
                 kernel_size=3, upsample_factor=2, interpolation_mode="linear", interpolation_align_corners=True):
        super(Decoder1D, self).__init__()
        self.layers = nn.ModuleList()
        self.conv1s = nn.ModuleList()
        self.output_features = output_features
        self.interpolation_mode = interpolation_mode
        self.interpolation_align_corners = interpolation_align_corners
        self.upsample_factor = upsample_factor
        in_channels = input_features
        for n_blocks, out_channels in zip(layer_blocks, layer_channels):
            layer = nn.ModuleList()
            self.conv1s.append(nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=1,
                                         stride=1))
            for i_block in range(n_blocks):
                layer.append(block(in_channels=out_channels, channels=out_channels, kernel_size=kernel_size, stride=1))
            in_channels = out_channels
            self.layers.append(layer)

    def forward(self, x):
        for (layer, conv1) in zip(self.layers, self.conv1s):
            print(x.shape)
            x = nn.functional.interpolate(x,
                                          size=(x.shape[-1] * self.upsample_factor),
                                          mode=self.interpolation_mode,
                                          align_corners=self.interpolation_align_corners)
            print(x.shape)
            x = conv1(x)
            for block in layer:
                print(x.shape)
                x = block(x)
        return nn.functional.interpolate(x,
                                         size=(self.output_features,),
                                         mode=self.interpolation_mode,
                                         align_corners=self.interpolation_align_corners)
