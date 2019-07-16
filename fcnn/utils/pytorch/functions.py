from torch.nn.functional import l1_loss, mse_loss
import torch


def regularized_loss(predicted, vae_x, mu, logvar, x, y, pred_loss=l1_loss, vae_loss=mse_loss, vae_weight=0.1,
                      kl_weight=0.1):
    loss_pred = pred_loss(predicted, y)
    loss_vae = vae_loss(vae_x, x)
    N = x.numel()/x.shape[0]
    loss_kl = (1 / N) * torch.sum(mu.pow(2) + logvar.exp() - logvar - 1)
    return loss_pred + (vae_weight * loss_vae) + (kl_weight * loss_kl)


