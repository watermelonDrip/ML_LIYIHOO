
<!-- https://www.heywhale.com/mw/project/61133875aca2460017a464a5 -->

# Homework 3: Image Classification
## Object
1. Solve image classification with convolutional neural networks.
2. Improve the performance with data augmentations.
3. Understand how to utilize unlabeled data and how it benefits.

## Description & Task
The images are collected from the food-11 dataset classified into 11 classes.
● Training set: 280 * 11 labeled images + 6786 unlabeled images
● Validation set: 60 * 11 labeled images
● Testing set: 3347 images

## Data
food-11 压缩包里有testing ,traning, validation 三个文件夹。
training 里面有两个文件夹labeled 和unlabeled。其中labeled里面有11个子文件夹分别名字从00到10，11个子文件夹。
每一个子文件夹代表一个class。比如00文件夹里的.jpg文件，名字为0_1.jpg，0_110.jpg,等。
这是官方网站要求的,如下：

CLASStorchvision.datasets.DatasetFolder(root: str, loader: Callable[[str], Any], extensions: Union[Tuple[str, ...], NoneType] = None, transform: Union[Callable, NoneType] = None, target_transform: Union[Callable, NoneType] = None, is_valid_file: Union[Callable[[str], bool], NoneType] = None) → None[SOURCE]
A generic data loader where the samples are arranged in this way:
```
root/class_x/xxx.ext
root/class_x/xxy.ext
root/class_x/xxz.ext

root/class_y/123.ext
root/class_y/nsdf3.ext
root/class_y/asd932_.ext
```
###  pre-processing dataset

```python
# 告诉DataLoader 怎么加载数据
train_set = DatasetFolder("food-11/training/labeled", loader=lambda x: Image.open(x), extensions="jpg", transform=train_tfm)
```
train_set输出的是一个tuple,如下
```
Dataset DatasetFolder
    Number of datapoints: 3080
    Root location: food-11/training/labeled
    StandardTransform
Transform: Compose(
               Resize(size=(128, 128), interpolation=bilinear, max_size=None, antialias=None)
               ToTensor()
           )
/usr/local/lib/python3.7/dist-packages/torch/utils/data/dataloader.py:481: UserWarning: This DataLoader will create 8 worker processes in total. Our suggested max number of worker in current system is 2, which is smaller than what this DataLoader is going to create. Please be aware that excessive worker creation might get DataLoader running slow or even freeze, lower the worker number to avoid potential slowness/freeze if necessary.
  cpuset_checked))
  ```
  

```python
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
```
