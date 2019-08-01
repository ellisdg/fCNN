"""
Modified from:
https://github.com/nkolot/GraphCMR/blob/master/models/graph_cnn.py
"""


from __future__ import division

import torch
import torch.nn as nn

from .graph_cmr_layers import GraphResBlock, GraphLinear
from ..resnet import resnet_18


class GraphCNN(nn.Module):
    def __init__(self, ref_vertices, adjacency_matrix, num_layers=5, num_channels=256, output_features=3,
                 encoder=resnet_18, encoder_outputs=512):
        super(GraphCNN, self).__init__()
        self.adjacency_matrix = adjacency_matrix
        self.ref_vertices = ref_vertices
        self.encoder = encoder(n_outputs=encoder_outputs)
        self.encoder_outputs = encoder_outputs
        layers = [GraphLinear(3 + self.encoder_outputs, 2 * num_channels),
                  GraphResBlock(2 * num_channels, num_channels, adjacency_matrix)]
        for i in range(num_layers):
            layers.append(GraphResBlock(num_channels, num_channels, adjacency_matrix))
        self.shape = nn.Sequential(GraphResBlock(num_channels, 64, adjacency_matrix),
                                   GraphResBlock(64, 32, adjacency_matrix),
                                   nn.GroupNorm(32 // 8, 32),
                                   nn.ReLU(inplace=True),
                                   GraphLinear(32, output_features))
        self.gc = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass
        Inputs:
            x: size = (B, 3, 224, 224)
        Returns:
            Regressed (subsampled) non-parametric shape: size = (B, 1723, 3)
            Weak-perspective camera: size = (B, 3)
        """
        batch_size = x.shape[0]
        ref_vertices = self.ref_vertices[None, :, :].expand(batch_size, -1, -1)
        x = self.encoder(x)
        x = x.view(batch_size, self.encoder_outputs, 1).expand(-1, -1, ref_vertices.shape[-1])
        x = torch.cat([ref_vertices, x], dim=1)
        x = self.gc(x)
        shape = self.shape(x)
        return shape
