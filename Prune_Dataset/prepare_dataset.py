import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append("..") 
from data_augmentation import *

DATASET_NCLASSES = {
    'MNIST': 10,
    'FashionMNIST': 10,
    'EMNIST': 26,
    'KMNIST': 10,
    'USPS': 10,
    'CIFAR10': 10,
    'SVHN': 10,
    'STL10': 10,
    'LSUN': 10,
    'Tiny-Imagenet': 200
}
DATASET_SIZES = {
    'MNIST': (28, 28),
    'FashionMNIST': (28, 28),
    'EMNIST': (28, 28),
    'QMNIST': (28, 28),
    'KMNIST': (28, 28),
    'USPS': (16, 16),
    'SVHN': (32, 32),
    'CIFAR10': (32, 32),
    'STL10': (96, 96),
    'Tiny-Imagenet': (64, 64)
}
DATASET_NORMALIZATION = {
    'MNIST': ((0.1307, ), (0.3081, )),
    'USPS': ((0.1307, ), (0.3081, )),
    'FashionMNIST': ((0.1307, ), (0.3081, )),
    'QMNIST': ((0.1307, ), (0.3081, )),
    'EMNIST': ((0.1307, ), (0.3081, )),
    'KMNIST': ((0.1307, ), (0.3081, )),
    'ImageNet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'Tiny-Imagenet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'STL10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}

def prepare_dataloader(dataset_name,batchsize):
    transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
    Cutout(n_holes=1, length=16)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
    ])
    root = '../data/'
    if dataset_name == 'CIFAR10':
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batchsize, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=False, num_workers=4)
    elif dataset_name == 'CIFAR100':
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batchsize, shuffle=True, num_workers=4)
        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=False, num_workers=4)
    elif dataset_name == 'MNIST':
        transform_train = transforms.Compose([
                        torchvision.transforms.Grayscale(3),
                        transforms.ToTensor(),
                        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
                    ])
        trainset = torchvision.datasets.MNIST(
            root=root, train=True, download=False, transform=transform_train)
        trainloader = torch.utils.data.DataLoader(
            trainset, batch_size=batchsize, shuffle=True, num_workers=4)

        transform_test = transform_train 
        testset =  torchvision.datasets.MNIST(
            root=root, train=False, download=False, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=False, num_workers=4)
    return trainloader,testloader
