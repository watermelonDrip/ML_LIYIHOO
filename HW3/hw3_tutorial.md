
<!-- https://www.heywhale.com/mw/project/61133875aca2460017a464a5 -->

# Homework 3: Image Classification
## Object
1. Solve image classification with convolutional neural networks.
2. Improve the performance with data augmentations.
3. Understand how to utilize unlabeled data and how it benefits.

## CNN
CNN强大在于卷积层强大的特征提取能力，当然我们可以利用CNN将特征提取出来后，用全连接层或决策树、支持向量机等各种机器学习算法模型来进行分类。
卷积层可通过重复使用卷积核有效地表征局部空间，卷积核（过滤器f i l t e r filterfilter）通过卷积的计算结果（相似度）表示该卷积核和扫描过的图像块的灰色格子部分相吻合的个数——该值越大则说明越符合卷积核的偏好程度。
——卷积的结果矩阵为特征映射. 

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
###  pre-processing dataset（load the training data）
https://machinelearningknowledge.ai/pytorch-dataloader-tutorial-with-example/#What_is_DataLoader_in_PyTorch

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
  ```
  
### construct data loaders 
```python
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
```
Iterate the train_loader ,输出imgs,lables 

## Model
CNN layer

## Train
### supervised-learning

### semi-supervised-learning



