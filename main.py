from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from models import *

# Training settings
parser = argparse.ArgumentParser(description="PyTorch CIFAR Playground")
parser.add_argument("--batch-size", type=int, default=64, metavar="N",
                    help = "input batch size for training (default: 64)")
parser.add_argument("--test-batch-size", type=int, default=1000, metavar="N",
                    help="input batch size for testing (default: 1000)")
parser.add_argument("--epochs", type=int, default=10, metavar="N",
                    help="number of epochs to train (default: 10)")
parser.add_argument("--lr", type=float, default=0.01, metavar="LR",
                    help="learning rate (default: 0.01)")
parser.add_argument("--momentum", type=float, default=0.5, metavar="M",
                    help="SGD momentum (default: 0.5)")
parser.add_argument("--no-cuda", action="store_true", default=False,
                    help="disables CUDA training")
parser.add_argument("--seed", type=int, default=1, metavar="S",
                    help="random seed (default: 1)")
parser.add_argument("--log-interval", type=int, default=10, metavar="N",
                    help="how many batches to wait before logging training status")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Data
kwargs = {"num_workers": 1, "pin_memory": True} if args.cuda else {}
transform_train = transforms.Compose([
    transforms.RandomCrop(32,padding=4),
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
trainset = datasets.CIFAR10(root="/home/x-czh/data_set", train=True,
                            transform=transform_train, download=False)
testset = datasets.CIFAR10(root="/home/x-czh/data_set", train=False,
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

# Model
args.resume = False
if args.resume:
    print("==> Resuming from checkpoint..")
    assert os.path.isdir("checkpoint"), "Error: no checkpoint directory found!"
    checkpoint = torch.load("./checkpoint/ckpt.t7")
    net = checkpoint["net"]
    best_acc = checkpoint["acc"]
    start_epoch = checkpoint["epoch"]
else:
    print("==> Building model..")
    model = LeNet()

if args.cuda:
    model.cuda()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

# Training
def train(epoch):
    model.train() # sets the module in training mode
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad() # zeroes the gradient buffers of all parameters
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
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

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

for epoch in range(1, args.epochs + 1):
    train(epoch)
    test()
