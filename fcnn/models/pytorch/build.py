import os

import torch

from .classification import resnet
from .classification import custom
from .segmentation import unet
from . import graph


def fetch_model_by_name(model_name, *args, **kwargs):
    try:
        if "res" in model_name.lower():
            return getattr(resnet, model_name)(*args, **kwargs)
        elif "graph" in model_name.lower():
            return getattr(graph, model_name)(*args, **kwargs)
        elif "unet" in model_name.lower():
            return getattr(unet, model_name)(*args, **kwargs)
        else:
            return getattr(custom, model_name)(*args, **kwargs)
    except AttributeError:
        raise ValueError("model name {} not supported".format(model_name))


def build_or_load_model(model_name, model_filename, n_features, n_outputs, n_gpus=0, bias=None, freeze_bias=False,
                        **kwargs):
    model = fetch_model_by_name(model_name, n_features=n_features, n_outputs=n_outputs, **kwargs)
    if bias is not None:
        model.fc.bias = torch.nn.Parameter(torch.from_numpy(bias))
    if freeze_bias:
        model.fc.bias.requires_grad_(False)
    if n_gpus > 1:
        model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    elif n_gpus > 0:
        model = model.cuda()
    if os.path.exists(model_filename):
        try:
            model.load_state_dict(torch.load(model_filename))
        except RuntimeError as error:
            if n_gpus > 1:
                model.module.load_state_dict(torch.load(model_filename))
            else:
                raise error
    return model



