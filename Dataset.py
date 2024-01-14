from torch.utils.data import Dataset
from PIL import Image
import torchvision
import torchvision.transforms as transforms
import sys
sys.path.append('..')
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
    # 'CIFAR10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR10': ((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    'CIFAR100': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'STL10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}

class CifarDataset(Dataset):
    def __init__(self,data,transform=None,transform_target=None):
        super().__init__()
        self.data = data
        self.transform = transform
        self.transform_target = transform_target
    
    def __getitem__(self, index):
        assert index < len(self.data)
        img, target = self.data[index][0],self.data[index][1]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.transform_target is not None:
            target = self.transform_target(target)
        return img, target
    def __len__(self):
        return len(self.data)

class Cifar(Dataset):
    def __init__(self,data,targets,transform=None,transform_target=None):
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.transform_target = transform_target
    
    def __getitem__(self, index):
        assert index < len(self.data)
        img, target = self.data[index],self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        if self.transform_target is not None:
            target = self.transform_target(target)
        return img, target
    def __len__(self):
        return len(self.data)
class TinyImageNet(Dataset):
    def __init__(self,data,targets,transform=None,transform_target=None):
        super().__init__()
        self.data = data
        self.targets = targets
        self.transform = transform
        self.transform_target = transform_target
    
    def __getitem__(self, index):
        assert index < len(self.data)
        img, target = self.data[index],self.targets[index]
        # C,W,H to W,H,C
        # print(img)
        # img = img.transpose(1,2,0)
        # img = Image.fromarray(np.uint8(img))
        if self.transform is not None:
            img = self.transform(img)
        if self.transform_target is not None:
            target = self.transform_target(target)
        return img, target
    def __len__(self):
        return len(self.data)


def prepare_dataset(dataset_name,batchsize,mask_file):
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
    ])
    root = './data/'
    if dataset_name == 'CIFAR10':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
            Cutout(n_holes=1, length=16)
            ])
        trainset = torchvision.datasets.CIFAR10(
            root=root, train=True, download=True, transform=transform_train)
        selected_data = torch.load('./Pruning_Results/{}/{}.pt'.format(dataset_name,mask_file))
        reserve = selected_data['selected_data'].cpu()
        s = sum(reserve)
        reserve_index = np.argwhere(np.array(reserve) == 1)
        train_reserve = np.array(trainset.data)[reserve_index.flatten()] 
        target_reserve = np.array(trainset.targets)[reserve_index.flatten()]

        # # all data
        # train_reserve = np.array(trainset.data)
        # target_reserve = np.array(trainset.targets)

        cifar_dataset = Cifar(train_reserve,target_reserve,transform_train)
        trainloader = torch.utils.data.DataLoader(
        cifar_dataset, batch_size=batchsize, shuffle=True, num_workers=2)

        testset = torchvision.datasets.CIFAR10(
            root=root, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=False, num_workers=4)
    elif dataset_name == 'CIFAR100':
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]),
        ])
        trainset = torchvision.datasets.CIFAR100(
            root=root, train=True, download=True, transform=transform_train)
        selected_data = torch.load('./Pruning_Results/{}/{}.pt'.format(dataset_name,mask_file))
        reserve = selected_data['selected_data'].cpu()
        reserve_index = np.argwhere(np.array(reserve) == 1)
        train_reserve = np.array(trainset.data)[reserve_index.flatten()]
        target_reserve = np.array(trainset.targets)[reserve_index.flatten()]
        cifar_dataset = Cifar(train_reserve,target_reserve,transform_train)
        trainloader = torch.utils.data.DataLoader(
        cifar_dataset, batch_size=batchsize, shuffle=True, num_workers=2)
        testset = torchvision.datasets.CIFAR100(
            root=root, train=False, download=True, transform=transform_test)
        testloader = torch.utils.data.DataLoader(
            testset, batch_size=batchsize, shuffle=False, num_workers=4)
    return trainloader,testloader