## The Pytorch implementation code of AdaPruner on CIFAR10/100 

#### Requirements

* python >= 3.6
* PyTorch >= 1.1.0
* Torch vision >= 0.3.0

#### Datasets

[CIFAR10]: http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
[CIFAR100]: http://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
[CIFAR10]

[CIFAR100]

#### Train DatasetPruner for pruned datasets

 ```--gpu```, the GPU used, the default is 0

```--model```, the model used, the default is resnet18

```--dataset```, the dataset used, the default is CIFAR10

```--compression_rate```, the expected compression rate, default is 0.9 (90%)

```--resume```, specify to use the warm-up mechanism


#### Example
Training mask index with 90% compression ratio on ResNet18
```python
python main.py --epoch 200 --gpu '0,1' --dataset CIFAR10 --model ResNet18 --compression_rate 0.90 --resume
```

#### Notice
Before training, please put pre-trained models at '''./checkpoint/CIFAR10/''', which is used to simplify the process of warm-up mechanism.
In this way, the warm-up mechanism will be conduct only once when pruning datasets with different compression ratios.
