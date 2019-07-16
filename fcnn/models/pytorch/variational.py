import torch.nn as nn
import torch


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
