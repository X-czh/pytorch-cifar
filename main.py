import argparse
import os
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torchvision import datasets, transforms

from utils import model_dict
from utils import parse_model_name
from utils import parse_milestones
from utils import adjust_learning_rate
from utils import get_current_time
from utils import AverageMeter

# device to use
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR Playground')
parser.add_argument('--data-path', type=str, default='./data', metavar='PATH',
                    help='data path (default: ./data)')
parser.add_argument('--num-classes', type=int, choices=[10, 100], default=10, metavar='N',
                    help='choose between 10/100')
parser.add_argument('--model', type=str, default='resnet-20', metavar='MODEL',
                    help='model architecture (default: resnet-20)')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--test-batch-size', type=int, default=128, metavar='N',
                    help='input batch size for testing (default: 128)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--milestones', type=str, default='0', metavar='M',
                    help='milestones to adjust learning rate, ints split by "-" (default: "0")')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--weight-decay', type=float, default=1e-4, metavar='W',
                    help='weight decay (default: 1e-4)')
parser.add_argument('--workers', type=int, default=4, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=75, metavar='S',
                    help='random seed (default: 75)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from latest checkpoint')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')

def get_data_loader(args):
    print('==> Preparing Data..')
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

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

    if args.num_classes == 10:
        trainset = datasets.CIFAR10(root=args.data_path, train=True,
                                    transform=transform_train, download=True)
        testset = datasets.CIFAR10(root=args.data_path, train=False,
                                   transform=transform_test, download=True)
    else:
        trainset = datasets.CIFAR100(root=args.data_path, train=True,
                                     transform=transform_train, download=True)
        testset = datasets.CIFAR100(root=args.data_path, train=False,
                                    transform=transform_test, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.cuda
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=args.cuda
    )

    return train_loader, test_loader

def get_model(args):
    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load('./checkpoint/ckpt.pth')
        model = checkpoint['model']
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']
    else:
        print('==> Building model..')
        arch, depth = parse_model_name(args.model)
        if depth != 0:
            model = model_dict[arch](depth=depth, num_classes=args.num_classes)
        else:
            model = model_dict[arch](num_classes=args.num_classes)
        best_acc = 0
        start_epoch = 1

    model.to(device)
    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True

    return model, best_acc, start_epoch

def get_criterion(args):
    criterion = nn.CrossEntropyLoss().to(device)
    return criterion

def get_optimizer(args, model):
    optimizer = optim.SGD(
        params=model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    return optimizer

def train(args, train_loader, model, criterion, optimizer, epoch, progress, train_time):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    model.train() # sets the module in training mode
    correct = 0

    end = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        # Measure data loading time
        data_time.update(time.time() - end)

        data, target = data.to(device), target.to(device)
        optimizer.zero_grad() # zeroes the gradient buffers of all parameters
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1)[1]
        correct += pred.eq(target.data).sum().item()

        # Measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # Print log
        if (batch_idx + 1) % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    train_time.update(batch_time.get_sum())

    # Save progress
    train_acc = 100. * correct / len(train_loader.dataset)
    progress['train'].append((epoch, loss.item(), train_acc,
                              batch_time.get_sum(), batch_time.get_avg(),
                              data_time.get_sum(), data_time.get_avg()))

def test(args, test_loader, model, criterion, epoch, progress, best_acc, test_time):
    model.eval() # sets the module in evaluation mode
    test_loss = 0
    correct = 0

    end = time.time()
    with torch.no_grad(): # disables gradient calculation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).sum().item()
    test_time.update(time.time() - end)

    # Print and save progress
    test_loss /= len(test_loader.dataset)
    test_acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), test_acc))
    progress['test'].append((epoch, test_loss, test_acc))

    # Save checkpoint
    if test_acc > best_acc:
        print('Saving checkpoint..')
        state = {
            'model': model.module if args.cuda else model,
            'acc': test_acc,
            'epoch': epoch + 1
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = test_acc

if __name__ == '__main__':
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.no_cuda:
        device = torch.device('cpu')
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    print('==> Training Settings:')
    for arg in vars(args):
        print(arg, getattr(args, arg))
    
    # Set data loader, model, criteriron, and optimizer
    train_loader, test_loader = get_data_loader(args)
    model, best_acc, start_epoch = get_model(args)
    criterion = get_criterion(args)
    optimizer = get_optimizer(args, model)
    lr = args.lr
    milestones = parse_milestones(args.milestones)

    # Train and record progress
    progress = {}
    progress['train'] = []
    progress['test'] = []
    train_time = AverageMeter()
    test_time = AverageMeter()

    print('==> Start training..')
    for epoch in range(start_epoch, start_epoch + args.epochs):
        adjust_learning_rate(optimizer, lr, epoch, milestones)
        train(args, train_loader, model, criterion, optimizer, epoch, progress, train_time)
        test(args, test_loader, model, criterion, epoch, progress, best_acc, test_time)

    progress['train_time'] = (train_time.get_avg(), train_time.get_sum())
        # record average epoch time and total training time
    progress['test_time'] = (test_time.get_avg() / len(test_loader.dataset), test_time.get_avg())
        # record average test time per image and average test time per test_loader.dataset

    # Save progress
    import pickle

    current_time = get_current_time()
    pickle.dump(progress, open('./' + args.model + ('-resume' if args.resume else '') +
                               '_progress_' + current_time + '.pkl', 'wb'))
