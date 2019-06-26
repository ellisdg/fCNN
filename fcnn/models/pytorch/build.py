from . import resnet


def fetch_model_by_name(model_name, *args, **kwargs):
    if "res" in model_name:
        return getattr(resnet, model_name, *args, **kwargs)
    else:
        raise ValueError("model name {} not supported".format(model_name))
