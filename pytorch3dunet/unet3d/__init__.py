import importlib

from .models.DMFNet import DMFNet
from .model import UNet3D, ResidualUNet3D


def get_model(config):
    def _model_class(class_name):
        m = importlib.import_module('pytorch3dunet.unet3d')
        clazz = getattr(m, class_name)
        return clazz

    assert 'model' in config, 'Could not find model configuration'
    model_config = config['model']
    model_class = _model_class(model_config['name'])
    return model_class(**model_config)
