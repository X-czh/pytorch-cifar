"""
Helper functions and classes
"""
from __future__ import print_function
import datetime
from models import *

model_dict = {
    'mlp': MLP,
    'lenet': LeNet,
    'vgg': VGG,
    'resnet': ResNet,
    'resnetv2': ResNetV2,
    # 'resnext': ResNeXt,
    'densenet': DenseNet
}

def parse_model_name(model_name):
    """Parses model name and returns model"""
    if model_name.find('-') == -1:
        arch = model_name
        assert arch in model_dict, 'Error: model not found!'
        model = model_dict[arch]()
    else:
        assert len(model_name.split('-')) == 2, 'Error: model name invalid!'
        arch, depth = model_name.split('-')
        assert arch in model_dict, 'Error: model not found!'
        depth = int(depth)
        model = model_dict[arch](depth)
    return model

def parse_milestones(str):
    """Parses the milestones argument and returns a list of int"""
    try:
        milestones = [int(e) for e in str.split('-')]
    except:
        print('Error: invlid input for milestones!')
    return milestones

def adjust_learning_rate(optimizer, lr, epoch, milestones):
    """Sets the learning rate to the initial LR decayed by 10
    once the number of epoch reaches one of the milestones"""
    if epoch in milestones:
        lr = lr * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

def get_current_time():
    """Returns current time in the format of %yyyy-%mm-%dd_%hh-%mm-%ss"""
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
