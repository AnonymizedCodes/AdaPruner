from datetime import date
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn

from tqdm import tqdm
import os
import argparse
import models
from models import *

from prepare_dataset import *
import numpy as np
import random
import sys
sys.path.append("..") 
from data_augmentation import *
class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            if key == 'p' or key == 'off' or key == 'max_scale':
                value = float(value)
            elif key == 'length' or key == 'patch_size' or key == 'deg':
                value = int(value)
            getattr(namespace, self.dest)[key] = value

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--batchsize',type=int,default=256)
parser.add_argument("--src", type=str, default=None, help="")
parser.add_argument('--trg', type=str, default='basic', help='')
parser.add_argument('--aug_arg', nargs='*',action=ParseKwargs,default=None)
parser.add_argument("--dataset", type=str, default='CIFAR10', help="")
parser.add_argument("--epoch", type=int, default=10, help="")
parser.add_argument("--compression_rate", type=float, default=0.9, help="")
parser.add_argument('--model', type=str, default='ResNet18',help='')
parser.add_argument('--gpu',type=str,default='0,1')
args = parser.parse_args()
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

#################### coefficients ##################
seed = 1
setup_seed(seed)
if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
    model = 'cifar_'+args.model
    if args.model == 'ResNet18':
        PRETRAINED_MODEL = 'all_data_pretrain_model_Resnet18'
    elif args.model == 'ResNet50':
        PRETRAINED_MODEL = 'all_data_pretrain_model_Resnet50'
model = args.model
if args.dataset == 'CIFAR10' or args.dataset == 'CIFAR100':
    model = 'cifar_'+args.model
compression_lambda = 50.0
compression_rate = args.compression_rate
THRESHOLD = 0.02
C = 1
GAMMA = 50
COLD_START = 0
# scale factor range:
scale_factor_start = 0.1
scale_factor_stop = 5
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
trainloader, testloader = prepare_dataloader(args.dataset, args.batchsize)

# Model
print('==> Building model..')
net = getattr(models,model)(batchsize=args.batchsize, training_num=DATASET_NUM[args.dataset], num_classes=DATASET_NCLASSES[args.dataset])

net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net,device_ids=[0,1])
    cudnn.benchmark = True
m = net.module.mask
m.requires_grad = False
if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load('./checkpoint/{}/{}.pth'.format(args.dataset,PRETRAINED_MODEL))

    model_dict = net.state_dict()
    pretrained_dict = {k:v for k, v in checkpoint['net'].items() if  k != 'module.mask'}
    net_dict = net.state_dict()
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    # net.load_state_dict(checkpoint['net'])
    best_acc = checkpoint['acc']
    print('Acc:',best_acc, checkpoint['epoch'])
    
 

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
sigmoid = nn.Sigmoid()
true_compression_rate = 1.0
scale_factor_step = (scale_factor_stop-scale_factor_start)/ ((int(args.epoch)-COLD_START)*len(trainloader))
scale_factor = scale_factor_start
tmp_cr = 1.0
# Training
def train(epoch):
    global scale_factor_list, compression_lambda, scale_factor, true_compression_rate, tmp_cr
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    num_of_iteration_below_CR = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if tmp_cr >= compression_rate:
            scale_factor += scale_factor_step
        mask_index = torch.arange(args.batchsize*batch_idx, args.batchsize*batch_idx + inputs.shape[0])
        inputs, targets, mask_index = inputs.to(device), targets.to(device), mask_index.to(device)
        optimizer.zero_grad()
        outputs, crossloss, select_loss, crossEntropy, mask = net(x=inputs,ground_truth=targets,mask_index=mask_index,scale_factor=scale_factor)
        crossloss = torch.mean(crossloss)
        select_loss = torch.mean(select_loss)
        loss = crossloss + select_loss + compression_lambda * (sigmoid(scale_factor*mask).norm(1)/mask.shape[0]-compression_rate) ** 2
        loss.backward()
        optimizer.step()

        tmp_cr = torch.where(sigmoid(scale_factor*mask)>0.5)[0].shape[0]/mask.shape[0]
        if tmp_cr < compression_rate:
            num_of_iteration_below_CR += 1
        if epoch == args.epoch - 1: 
            outside_ratio = (torch.where(sigmoid(scale_factor*mask)<0.1)[0].shape[0] + torch.where(sigmoid(scale_factor*mask)>0.9)[0].shape[0])/mask.shape[0]
            if outside_ratio < 0.9:
                scale_factor += scale_factor_step * 10

        if epoch < COLD_START:
            compression_lambda = 0
        else:
            compression_lambda = C + GAMMA * np.abs(tmp_cr - compression_rate)
        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
        
    if num_of_iteration_below_CR >= len(trainloader)/16:
        print('=========Slow Down the Compression==========')
        scale_factor = max(0, scale_factor - scale_factor_step * (num_of_iteration_below_CR)) 
    # TODO：
    true_compression_rate = torch.where(sigmoid(scale_factor*mask)>0.5)[0].shape[0]/mask.shape[0]
    return scale_factor, true_compression_rate

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            mask_index = torch.arange(args.batchsize*batch_idx, args.batchsize*batch_idx + inputs.shape[0])
            outputs, ____, ___, __, _ = net(x=inputs,ground_truth=targets,mask_index=mask_index,scale_factor=1.0)
            loss = criterion(outputs, targets)
            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
 
    # Save checkpoint.
    acc = 100.*correct/total
    if acc > best_acc:
        best_acc = acc
        print('Best Accuracy:{}%'.format(best_acc))
    return acc
 
def save_mask(mask_value,scale_factor,filename,state_parameter_1,state_parameter_2):
    selected_data = sigmoid(scale_factor*mask_value)
    zero = torch.zeros_like(selected_data)
    one = torch.ones_like(selected_data)
    selected_data = torch.where(selected_data > 0.5,one,zero)
    state = {
        'epoch': state_parameter_1,
        'acc': state_parameter_2,
        'selected_data': selected_data,
    }
    torch.save(state,'./results/{}/{}.pt'.format(args.dataset,filename))

cr_best_acc = 0.
for epoch in tqdm(range(start_epoch, start_epoch+args.epoch)):
 
    if not m.requires_grad and epoch >= COLD_START:
        m.requires_grad = True
    scale_factor, true_compression_rate = train(epoch)
    test_acc = test(epoch)
    scheduler.step()
    key = 'module.mask'
    mask_value = net.state_dict().get(key)
    cur_compression_rate = torch.where(sigmoid(scale_factor*mask_value)>0.5)[0].shape[0]/mask_value.shape[0]
    if np.abs(compression_rate - cur_compression_rate) < THRESHOLD and (test_acc > cr_best_acc):
        print('Best Mask:',cur_compression_rate,compression_rate)
        cr_best_acc = test_acc
        save_mask(mask_value,
                    scale_factor,
                    'best_mask_pretrained_{}_{}_{}'.format(str(int(compression_rate*100)),
                    date.today().strftime("%m%d"),model),
                    epoch-start_epoch,
                    test_acc)
    print(true_compression_rate, torch.where(sigmoid(scale_factor*mask_value)>0.5)[0].shape[0]/mask_value.shape[0])
    save_mask(mask_value, scale_factor,'final_mask_pretrained_{}_{}_{}'.format(str(int(compression_rate*100)),date.today().strftime("%m%d"),model),epoch-start_epoch,test_acc)
