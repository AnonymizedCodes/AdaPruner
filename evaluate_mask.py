'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
from Dataset import  prepare_dataset
import os
import argparse
import random
import numpy as np
from tqdm import tqdm
import models
import sys
sys.path.append("..") 
from data_augmentation import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batch',type=int,default=256)
parser.add_argument("--dataset", type=str, default='CIFAR10', help="")
parser.add_argument('--file_name',type=str,default='')
parser.add_argument('--model',type=str,default='ResNet18')
parser.add_argument('--gpu',type=str,default='0,1')
parser.add_argument('--seed',type=int,default=1000)
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
#################### coefficients ##################
seed = args.seed
setup_seed(seed)
mask_file = args.file_name
model = args.model
####################################################
DATASET_NCLASSES = {
    'MNIST': 10,
    'FashionMNIST': 10,
    'EMNIST': 26,
    'KMNIST': 10,
    'USPS': 10,
    'CIFAR10': 10,
    'CIFAR100': 100,
    'SVHN': 10,
    'STL10': 10,
    'LSUN': 10,
    'Tiny-Imagenet': 200
}
DATASET_NUM = {
    'MNIST': 60000,
    'CIFAR10': 50000,
    'CIFAR100': 50000,
    'Tiny-Imagenet': 100000
}
device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
trainloader, testloader = prepare_dataset(args.dataset, args.batch,mask_file)

# Model
print('==> Building model..')
net = getattr(models,model)(batchsize=args.batch, training_num=DATASET_NUM[args.dataset], class_num=DATASET_NCLASSES[args.dataset])
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net,device_ids=[0,1])
    # cudnn.benchmark = True

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/all_data_pretrain_model_Resnet50.pth')
    net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch']

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc

    print('Best Acc:', best_acc)

if __name__ == '__main__':
    for epoch in tqdm(range(start_epoch, start_epoch+200)):
        train(epoch)
        test(epoch)
        scheduler.step()
 
