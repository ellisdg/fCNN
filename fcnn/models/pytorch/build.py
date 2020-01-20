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
