from __future__ import print_function, division
import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import *

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Playground')
parser.add_argument('--path', type=str, default='/home/x-czh/data_set', metavar='PATH',
                    help='data path (default: /home/x-czh/data_set)')
# parser.add_argument('--model', type=str, default='lenet', metavar='MODEL',
                    # help='model architecture (default: lenet)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help = 'input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--workers', type=int, default=2, metavar='N',
                    help='number of data loading workers (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--resume', type=bool, default=False, metavar='T/F',
                    help='resume from latest checkpoint (default: False)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')

def get_data_loader(args):
    kwargs = {'num_workers': args.workers, 'pin_memory': True} if args.cuda else {}
    
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                             std=(0.2023, 0.1994, 0.2010))
    ])
    trainset = datasets.CIFAR10(root=args.path, train=True,
                                transform=transform_train, download=False)
    testset = datasets.CIFAR10(root=args.path, train=False,
                               transform=transform_test, download=False)
    train_loader = torch.utils.data.DataLoader(
        dataset=trainset, 
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=True,
        **kwargs
    )

    return train_loader, test_loader

def get_model(args):
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.t7')
        model = checkpoint['model']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        # model = MLP()
        # model = LeNet()
        # model = VGG('VGG19')
        model = ResNet20()
        best_acc = 0
        start_epoch = 1

    if args.cuda:
        model.cuda()
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    return model, best_acc, start_epoch

def get_criterion(args):
    criterion = nn.CrossEntropyLoss()
    if args.cuda:
        criterion.cuda()
    return criterion

def get_optimizer(args, model):
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    return optimizer

def train(args, train_loader, model, criterion, optimizer, epoch, progress):
    model.train() # sets the module in training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        correct = 0
        train_acc = 0
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad() # zeroes the gradient buffers of all parameters
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1]
        correct = pred.eq(target.data).cpu().sum()
        train_acc = 100. * correct / len(data)

        # Print and save progress
        partial_epoch = epoch + batch_idx / len(train_loader) - 1
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))
        progress['train'].append((partial_epoch, loss.data[0], train_acc))

def test(args, test_loader, model, criterion, epoch, progress, best_acc):
    model.eval() # sets the module in evaluation mode
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += criterion(output, target).data[0]
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).cpu().sum()

    # Print and save progress
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset) 
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    progress['test'].append((epoch, test_loss, test_acc))

    # Save checkpoint
    if test_acc > best_acc:
        print('Saving..')
        state = {
            'model': model.module if args.cuda else model,
            'acc': test_acc,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.t7')
        best_acc = test_acc

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    # Set the seed for generating random numbers
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    # Set data loader, model, criteriron, and optimizer
    train_loader, test_loader = get_data_loader(args)
    model, best_acc, start_epoch = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model)

    # Train
    progress = {}
    progress['train'] = []
    progress['test'] = []

    for epoch in range(start_epoch, args.epochs + 1):
        train(args, train_loader, model, criterion, optimizer, epoch, progress)
        test(args, test_loader, model, criterion, epoch, progress, best_acc)
    
    # Save info
    import cPickle as pickle
    pickle.dump(progress, open('./progress.pkl','wb'))
