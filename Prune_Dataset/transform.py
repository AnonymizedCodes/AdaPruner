
import torchvision.transforms as transforms
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
    'tiny-ImageNet': 200
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
    'tiny-ImageNet': (64, 64)
}

DATASET_NORMALIZATION = {
    'MNIST': ((0.1307, ), (0.3081, )),
    'USPS': ((0.1307, ), (0.3081, )),
    'FashionMNIST': ((0.1307, ), (0.3081, )),
    'QMNIST': ((0.1307, ), (0.3081, )),
    'EMNIST': ((0.1307, ), (0.3081, )),
    'KMNIST': ((0.1307, ), (0.3081, )),
    'ImageNet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'tiny-ImageNet': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'CIFAR100': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    'STL10': ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
}


def make_transform(data_augmentation=None, dataset_name='MNIST', ratio=0, to3channels=False, args=None):
 
    transform_list=[]
    if data_augmentation == None:
        if dataset_name in ['MNIST', 'USPS'] and to3channels:
            transform_list.append(transforms.Grayscale(3))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        transform = transforms.Compose(transform_list)
        DA_name = 'baseline'
    elif data_augmentation == 'blur':
        transform_list.append(Blur(p=args['p']))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        DA_name = 'blur'+'_'+str(args['p'])
    elif data_augmentation == 'patchgaussian':
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        transform_list.append(AddPatchGaussian(args['patch_size'],args['max_scale']))
        DA_name = 'patchgaussian'+'_'+str(args['patch_size'])+'_'+str(args['max_scale'])
    elif data_augmentation == 'cutout':
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        transform_list.append(Cutout(1, args['length'], args['p']))
        DA_name = 'cutout'+'_'+str(args['length'])+'_'+str(args['p'])
    elif data_augmentation == 'equalize':
        transform_list.append(Equalize(p=args['p']))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        DA_name = 'equalize'+'_'+str(args['p'])
    elif data_augmentation == 'flipLR':
        transform_list.append(transforms.RandomHorizontalFlip(args['p']))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        DA_name = 'flipLR'+'_'+str(args['p'])
    elif data_augmentation == 'flipUD':
        transform_list.append(transforms.RandomVerticalFlip(args['p']))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        DA_name = 'flipUD'+'_'+str(args['p'])
    elif data_augmentation == 'rotate':
        transform_list.append(Rotate(deg=args['deg'],p=args['p'],mode=args['mode']))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        DA_name = 'rotate'+'_'+str(args['deg'])+'_'+str(args['p'])+'_'+args['mode']
    elif data_augmentation == 'shearX':
        transform_list.append(ShearX(p=args['p'],off=args['off'],mode=args['mode']))
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        DA_name = 'shearX'+'_'+str(args['p'])+'_'+str(args['off']) +'_'+args['mode']

    else:  #basic
        if dataset_name in ['MNIST', 'USPS'] and to3channels:
            transform_list.append(transforms.Grayscale(3))
        transform_list.append(transforms.RandomCrop(DATASET_SIZES[dataset_name][0], padding=4))
        transform_list.append(transforms.RandomHorizontalFlip())
        transform_list.append(transforms.ToTensor())
        transform_list.append(transforms.Normalize(*DATASET_NORMALIZATION[dataset_name]))
        transform = transforms.Compose(transform_list)
        DA_name = 'basic'
    transform = transforms.Compose(transform_list)
    
    return transform, DA_name