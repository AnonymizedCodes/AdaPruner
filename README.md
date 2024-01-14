## The Pytorch implementation of "Not All Data Matters: An End-to-End Adaptive Dataset Pruning Framework for Enhancing Model Performance and Efficiency". 

#### Requirements

* python >= 3.6
* PyTorch >= 1.1.0
* Torch vision >= 0.3.0

#### Datasets

[CIFAR10]: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
[CIFAR100]: http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
[Tiny-ImageNet]: http://cs231n.stanford.edu/tiny-imagenet-200.zip

**Note:** During runtime, the code will automatically download the dataset to the `data` folder.

####  Pruning Results
The pruning results of CIFAR10 and CIFAR100 can be found at '''./Pruning_Results/'''.

Researchers can directly use it to construct the pruned training set in an offline manner.

#### Using Pruned Datasets to Train models
 ```--gpu```, the GPU used, the default is 0

```--model```, the model used, the default is resnet18

```--file_name```, the pruning results used

```--seed```, the random seed used for multiple runs

#### Example
Training ResNet18 using pruned dataset with 90% compression ratio 
``` python evaluate_mask.py --file_name 'mask_index_90' --model 'ResNet18' --dataset 'CIFAR10' --gpu '0,1' --seed 10```

**To prune datasets with expected compression ratios, please refer to README in './Prune_Dataset/'**
