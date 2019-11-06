from functools import partial

import numpy as np
import torch.nn as nn
import torch

from fcnn.models.pytorch.decoder import MyronenkoDecoder, MirroredDecoder
from fcnn.models.pytorch.myronenko import MyronenkoEncoder, MyronenkoConvolutionBlock
from fcnn.models.pytorch.resnet import conv1x1x1


class VariationalBlock(nn.Module):
    def __init__(self, in_size, n_features, out_size, return_parameters=False):
        super(VariationalBlock, self).__init__()
        self.n_features = n_features
        self.return_parameters = return_parameters
        self.dense1 = nn.Linear(in_size, out_features=n_features*2)
        self.dense2 = nn.Linear(self.n_features, out_size)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        _x = x
        _x = self.dense1(_x)
        mu, logvar = torch.split(_x, self.n_features, dim=1)
        z = self.reparameterize(mu, logvar)
        out = self.dense2(z)
        if self.return_parameters:
            return out, mu, logvar, z
        else:
            return out, mu, logvar


class ConvolutionalAutoEncoder(nn.Module):
    def __init__(self, input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear", encoder=MyronenkoEncoder,
                 decoder=MyronenkoDecoder, n_outputs=None, layer_widths=None, decoder_mirrors_encoder=False):
        super(ConvolutionalAutoEncoder, self).__init__()
        if encoder_blocks is None:
            encoder_blocks = [1, 2, 2, 4]
        self.encoder = encoder(n_features=n_features, base_width=base_width, layer_blocks=encoder_blocks,
                               feature_dilation=feature_dilation, downsampling_stride=downsampling_stride,
                               layer_widths=layer_widths)
        if decoder_mirrors_encoder:
            decoder_blocks = encoder_blocks
            decoder = MirroredDecoder
        if decoder_blocks is None:
            decoder_blocks = [1] * (len(encoder_blocks) - 1)
        self.decoder = decoder(base_width=base_width, layer_blocks=decoder_blocks,
                               upsampling_scale=downsampling_stride, feature_reduction_scale=feature_dilation,
                               upsampling_mode=interpolation_mode, layer_widths=layer_widths)
        self.final_convolution = conv1x1x1(in_planes=base_width, out_planes=n_features, stride=1)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        return x


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
        x = self.in_conv(x).flatten(start_dim=1)
        x, mu, logvar = self.var_block(x)
        x = self.relu(x).view(-1, *self.reduced_shape)
        x = self.out_conv(x)
        x = self.upsample(x)
        return x, mu, logvar


class VariationalAutoEncoder(ConvolutionalAutoEncoder):
    def __init__(self, n_reduced_latent_feature_maps=16, vae_features=128, variational_layer=MyronenkoVariationalLayer,
                 input_shape=None, n_features=1, base_width=32, encoder_blocks=None, decoder_blocks=None,
                 feature_dilation=2, downsampling_stride=2, interpolation_mode="trilinear", encoder=MyronenkoEncoder,
                 decoder=MyronenkoDecoder, n_outputs=None, layer_widths=None, decoder_mirrors_encoder=False):
        super(VariationalAutoEncoder, self).__init__(input_shape=input_shape, n_features=n_features,
                                                     base_width=base_width, encoder_blocks=encoder_blocks,
                                                     decoder_blocks=decoder_blocks, feature_dilation=feature_dilation,
                                                     downsampling_stride=downsampling_stride,
                                                     interpolation_mode=interpolation_mode, encoder=encoder,
                                                     decoder=decoder, n_outputs=n_outputs, layer_widths=layer_widths,
                                                     decoder_mirrors_encoder=decoder_mirrors_encoder)
        if vae_features is not None:
            depth = len(encoder_blocks) - 1
            n_latent_feature_maps = base_width * (feature_dilation ** depth)
            latent_image_shape = np.divide(input_shape, downsampling_stride ** depth)
            self.var_layer = variational_layer(in_features=n_latent_feature_maps,
                                               input_shape=latent_image_shape,
                                               reduced_features=n_reduced_latent_feature_maps,
                                               latent_features=vae_features,
                                               upsampling_mode=interpolation_mode)

    def forward(self, x):
        x = self.encoder(x)
        x, mu, logvar = self.var_layer(x)
        x = self.decoder(x)
        x = self.final_convolution(x)
        return x, mu, logvar
