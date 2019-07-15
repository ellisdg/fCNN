from . import resnet
from . import custom


def fetch_model_by_name(model_name, *args, **kwargs):
    try:
        if "res" in model_name:
            return getattr(resnet, model_name)(*args, **kwargs)
        else:
            return getattr(custom, model_name)(*args, **kwargs)
    except AttributeError:
        raise ValueError("model name {} not supported".format(model_name))
