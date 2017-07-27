"""
Helper functions
"""
import datetime
from models import *

model_dict = {
    'mlp': MLP(),
    'lenet': LeNet(),
    'vgg11': VGG('VGG11'),
    'vgg13': VGG('VGG13'),
    'vgg16': VGG('VGG16'),
    'vgg19': VGG('VGG19'),
    'resnet20': ResNet20(),
    'resnet32': ResNet32(),
    'resnet44': ResNet44(),
    'resnet56': ResNet56(),
    'resnet110': ResNet110()
}

def get_net(model_name):
    """ Return a net corresponding to the given model name """
    assert model_name in model_dict, 'Model not built in library'
    return model_dict[model_name]

def get_current_time():
    """ Return current time in the format of %yyyy-%mm-%dd_%hh-%mm-%ss """
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')
