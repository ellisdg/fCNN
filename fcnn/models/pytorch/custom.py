from torch import nn
import numpy as np

from .myronenko import MyronenkoEncoder, MyronenkoDecoder, MyronenkoVariationalLayer
from .resnet import conv1x1x1


class VariationalAutoEncoder(nn.Module):
    def __init__(self, input_shape, n_features, base_width=32, encoder_blocks=None, decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, n_reduced_latent_feature_maps=16, vae_features=128,
                 interpolation_mode="trilinear"):
        super(VariationalAutoEncoder, self).__init__()
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]
        self.vae_features = vae_features
        self.encoder = MyronenkoEncoder(base_width=base_width, layer_blocks=encoder_blocks,
                                        feature_dilation=feature_dilation, downsampling_stride=downsampling_stride)
        n_latent_feature_maps = base_width * (feature_dilation ** (len(encoder_blocks) - 1))
        self.latent_shape = (n_latent_feature_maps, *np.divide(input_shape, downsampling_stride))
        self.var_layer = MyronenkoVariationalLayer(in_features=n_latent_feature_maps, input_shape=self.latent_shape,
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
        _x = x
        _x = self.encoder(_x)
        _x, mu, logvar = self.var_layer(_x)
        _x = self.decoder(_x)
        _x = self.final_convolution(_x)
        return _x, mu, logvar


class RegularizedResNet(VariationalAutoEncoder):
    def __init__(self, n_outputs, *args, **kwargs):
        super(RegularizedResNet, self).__init__(*args, **kwargs)
        self.dense = nn.Linear(self.var_layer.in_size, n_outputs)

    def forward(self, x):
        latent_tensor = self.encoder(x)
        reduced_latent_vector = self.var_layer.in_conv(latent_tensor).flatten()
        parameters, mu, logvar = self.var_layer.var_block(reduced_latent_vector)
        _x = self.var_layer.relu(parameters).reshape(self.reduced_shape)
        _x = self.var_layer.out_conv(_x)
        _x = self.var_layer.upsample(_x)
        vae_output = self.decoder(parameters)
        output = self.dense(reduced_latent_vector)
        return output, vae_output, mu, logvar

