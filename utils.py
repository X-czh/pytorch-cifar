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
    'resnet110': ResNet110(),
    'resnet164': ResNet164(),
    'resnet1001': ResNet1001(),
    'densenet40': DenseNet40(),
    'densenet100': DenseNet100(),
    'densenet250': DenseNet250(),
    'densenet190': DenseNet190(),
    'densenet10':DenseNet10()
}

def get_net(model_name):
    """ Return a net corresponding to the given model name """
    assert model_name in model_dict, 'Model not built in library'
    return model_dict[model_name]

def get_current_time():
    """ Return current time in the format of %yyyy-%mm-%dd_%hh-%mm-%ss """
    now = datetime.datetime.now()
    return now.strftime('%Y-%m-%d_%H-%M-%S')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def get_sum(self):
        return self.sum

    def get_avg(self):
        return self.avg
