from torch import nn
import numpy as np

from .myronenko import MyronenkoEncoder, MyronenkoVariationalLayer
from .decoder import MyronenkoDecoder, BasicDecoder
from .resnet import conv1x1x1, ResNet, BasicBlock


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_shape, n_features, base_width=32, encoder_blocks=None, decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, n_reduced_latent_feature_maps=16, vae_features=128,
                 interpolation_mode="trilinear"):
        super(VariationalAutoEncoder, self).__init__()
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]
        self.vae_features = vae_features
        self.encoder = MyronenkoEncoder(n_features=n_features, base_width=base_width, layer_blocks=encoder_blocks,
                                        feature_dilation=feature_dilation, downsampling_stride=downsampling_stride)
        depth = len(encoder_blocks) - 1
        n_latent_feature_maps = base_width * (feature_dilation ** depth)
        self.latent_image_shape = np.divide(input_shape, downsampling_stride ** depth)
        self.var_layer = MyronenkoVariationalLayer(in_features=n_latent_feature_maps,
                                                   input_shape=self.latent_image_shape,
                                                   reduced_features=n_reduced_latent_feature_maps,
                                                   latent_features=self.vae_features,
                                                   upsampling_mode=interpolation_mode)
        if decoder_blocks is None:
            decoder_blocks = [1] * (len(encoder_blocks) - 1)
        self.decoder = MyronenkoDecoder(base_width=base_width, layer_blocks=decoder_blocks,
                                        upsampling_scale=downsampling_stride, feature_reduction_scale=feature_dilation,
                                        upsampling_mode=interpolation_mode)
        self.final_convolution = conv1x1x1(in_planes=base_width, out_planes=n_features, stride=1)

    def forward(self, x):
        x = self.encoder(x)
        x, mu, logvar = self.var_layer(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        return x, mu, logvar


class RegularizedResNet(VariationalAutoEncoder):
    def __init__(self, n_outputs, *args, **kwargs):
        super(RegularizedResNet, self).__init__(*args, **kwargs)
        self.fc = nn.Linear(self.var_layer.in_size, n_outputs)

    def forward(self, x):
        x = self.encoder(x)
        x = self.var_layer.in_conv(x).flatten(start_dim=1)
        output = self.fc(x)
        x, mu, logvar = self.var_layer.var_block(x)
        x = self.var_layer.relu(x).view(-1, *self.var_layer.reduced_shape)
        x = self.var_layer.out_conv(x)
        x = self.var_layer.upsample(x)
        x = self.decoder(x)
        vae_output = self.final_convolution(x)
        return output, vae_output, mu, logvar


class RegularizedBasicResNet(nn.Module):
    def __init__(self, n_features, upsampling_mode="trilinear", upsampling_scale=2, plane_dilation=2,
                 decoding_layers=None, latent_planes=512, layer_block=BasicBlock, **encoder_kwargs):
        super(RegularizedBasicResNet, self).__init__()
        if decoding_layers is None:
            decoding_layers = [1, 1, 1, 1, 1, 1, 1]
        self.encoder = _ResNetLatent(block=layer_block, n_features=n_features, **encoder_kwargs)
        self.decoder = BasicDecoder(upsampling_scale=upsampling_scale, upsampling_mode=upsampling_mode,
                                    plane_dilation=plane_dilation, layers=decoding_layers, in_planes=latent_planes,
                                    block=layer_block)
        out_decoder_planes = int(latent_planes/(plane_dilation**len(decoding_layers)))
        self.final_convolution = conv1x1x1(in_planes=out_decoder_planes, out_planes=n_features, stride=1)

    def forward(self, x):
        out, x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        return out, x


class _ResNetLatent(ResNet):
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        latent = x
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x, latent
