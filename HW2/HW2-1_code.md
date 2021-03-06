# HW2-1

## upload the new .ipynb, and run alll

## pass the simple baseline using the sample, we can see the results

## analysis Code

1. Preparing Data

```python
import numpy as np

print('Loading data ...')

data_root='./timit_11/'
train = np.load(data_root + 'train_11.npy')
train_label = np.load(data_root + 'train_label_11.npy')
test = np.load(data_root + 'test_11.npy')

print('Size of training data: {}'.format(train.shape))
print('Size of train_label: {}'.format(train_label.shape))
print('Size of testing data: {}'.format(test.shape))
```

```python
Loading data ...
Size of training data: (1229932, 429)
Size of train_label: (1229932, )
Size of testing data: (451552, 429)
```

- 给data_root配置一个地址
- np.load 是读数据的函数。给train 配置train_11.npy的文件，这个文件的地址在data_root里
- 给train_label配置一个train_label_11.npy文件，地址也在data_root里。
- 给test配置一个test_11.npy文件，地址也在data_root里。
- 可以看到，test 没给label 文件。所以我们需要通过训练得到一个model来给test 一个label文件。
1. Create Dataset

(1) 初始化

```python
import torch
from torch.utils.data import Dataset

class TIMITDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = torch.from_numpy(X).float()
        if y is not None: #train_set (with label y)
            y = y.astype(np.int)
            self.label = torch.LongTensor(y)
        else: #test_set( without label)
            self.label = None

    def __getitem__(self, idx): #每次怎么读数据(Map-style datasets)
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):#就是返回整个数据集的长度:
        return len(self.data)
```

- import torch
- torch.utils.data.Dataset是代表这一数据的抽象类
- 定义一个类TIMITDataset
- `def __init__(self, X, y=None):` 首先是对象初始化
- `torch.from_numpy(X).float()`  这个方法将numpy类转换成tensor类，
- 如果label 存在的话，`y = y.astype(np.int)`  复制y(array),cast to int类型
- `torch.LongTensor(y)` y转换成 Long类型的张量，配置给self.label

（2） 从labeled 数据里分配training set 和validation set， 通过VAL_RATIO 来调整比例  **(VAL_RATIO 可改)**

```python
VAL_RATIO = 0.2

percent = int(train.shape[0] * (1 - VAL_RATIO))
train_x, train_y, val_x, val_y = train[:percent], train_label[:percent], train[percent:], train_label[percent:]
print('Size of training set: {}'.format(train_x.shape))
print(train_x)
print(train_y)
print('Size of validation set: {}'.format(val_x.shape))
```

```python
Size of training set: (983945, 429)
[[-1.06118095 -0.82970196  1.03218007 ... -0.26369026 -0.41616243
   0.14449082]
 [-1.06031787 -0.86362785  1.22968948 ... -0.82504684 -1.04952157
  -0.49688351]
 [-1.06123281 -0.90362591  0.94239712 ...  0.60251135 -1.96513951
   0.01085626]
 ...
 [-0.45596942 -0.03308518  0.92863154 ... -0.50935656  0.25137249
  -0.05006642]
 [-0.19701307  0.36446738  0.8117668  ...  0.01844472  0.4166162
   0.72473669]
 [ 0.06918774  0.30106232  0.00551695 ...  0.12624711  0.54119331
   1.68067896]]
['36' '36' '36' ... '5' '5' '5']
Size of validation set: (245987, 429)
```

（3） 创建data loader 从data set。 BATCH_SIZE 设置batch的大小  **(BATCH_SIZE 可改)**

```python
BATCH_SIZE = 64

from torch.utils.data import DataLoader

train_set = TIMITDataset(train_x, train_y)
print(train_x)
val_set = TIMITDataset(val_x, val_y)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data
val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)
```

- `from torch.utils.data import DataLoader`  ,首先Pytorch的数据读取主要包含三个类:Dataset；DataLoader；DataLoaderIter。 这三者大致是一个一次封装的关系，1 装进2，2 装进3。
    
    [**Pytorch数据读取**](https://www.notion.so/Pytorch-3421c984a3344d3bab80b0f404eacb84)
    
- train_loader 和 val_loader 两行的类定义是一样的，我们放一起介绍。
    
    类定义为: `class torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, sampler=None, batch_sampler=None, num_workers=0, collate_fn=<function default_collate>, pin_memory=False, drop_last=False)`
    
    可以看到, 主要参数有这么几个:
    
    1. `dataset` : 即上面自定义的dataset.
    2. `collate_fn`: 这个函数用来打包batch, 后面详细讲.
    3. `num_worker`: 非常简单的多线程方法, 只要设置为>=1, 就可以多线程预读数据啦.

这个类其实就是下面将要讲的`DataLoaderIter`的一个框架, 一共干了两件事: 1.定义了一堆成员变量, 到时候赋给`DataLoaderIter`, 2.然后有一个`__iter__()` 函数, 把自己 "装进" `DataLoaderIter` 里面.

- train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) #only shuffle the training data

       1.  data set : train_set

       2.  batch_size=BATCH_SIZE(64)

       3. shuffle=True

- val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False)

       1.  data set : val_set

       2.  batch_size=BATCH_SIZE(64)

       3. shuffle=False

```python
[[-1.06118095 -0.82970196  1.03218007 ... -0.26369026 -0.41616243
   0.14449082]
 [-1.06031787 -0.86362785  1.22968948 ... -0.82504684 -1.04952157
  -0.49688351]
 [-1.06123281 -0.90362591  0.94239712 ...  0.60251135 -1.96513951
   0.01085626]
 ...
 [-0.45596942 -0.03308518  0.92863154 ... -0.50935656  0.25137249
  -0.05006642]
 [-0.19701307  0.36446738  0.8117668  ...  0.01844472  0.4166162
   0.72473669]
 [ 0.06918774  0.30106232  0.00551695 ...  0.12624711  0.54119331
   1.68067896]]
```

(4) 

Cleanup the unneeded variables to save memory.

**notes: if you need to use these variables later, then you may remove this block or clean up unneeded variables laterthe data size is quite huge, so be aware of memory usage in colab**

```python
import gc

del train, train_label, train_x, train_y, val_x, val_y
gc.collect() #避免重新启动笔记本计算机的方式。
```

1. create model

```python
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 39) 

        self.act_fn = nn.Sigmoid()

    def forward(self, x): #forward 流程
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
        
        return x
```

- 定义Classifier这个类,`Classifier(nn.Module)` ,Classifier 类继承nn.Module， 也就是说Classifier的父类是nn.Module。
- 初始化：`super(Classifier, self).__init__()`. Classifier 继承了nn.Module的所有属性和方法。`super(Classifier, self).init()`就是对继承自父类nn.Module的属性进行初始化。
    
    [self super](https://www.notion.so/self-super-1b7f793439894b5c9b184e347beb80f9)
    
- 从nn.Module继承过来的属性有：
    
    （1） **`[nn.Linear](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear)`**
    
    **Applies a linear transformation to the incoming data: $y = xA^T + b$**
    
    定义：  torch.nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)
    
    [torch.nn.Linear(input,output);](https://www.notion.so/torch-nn-Linear-input-output-c60da7fbc6cd4caab311660cd154939f)
    
    - 第一层
        
         `self.layer1 = nn.Linear(429, 1024)` 
        
        ![Untitled](https://user-images.githubusercontent.com/69283174/141734188-1d2f5e81-2ae2-4401-8490-28f83c7231d3.png)

        
    - 第二层        self.layer2 = nn.Linear(1024, 512)
    - 第三层       self.layer3 = nn.Linear(512, 128)
    - 输出           self.out = nn.Linear(128, 39)
    - 激活函数    self.act_fn = nn.Sigmoid()
    - 整个流程简单表示这样的
    
     ![Untitled 1](https://user-images.githubusercontent.com/69283174/141734203-df2017c8-da7d-4213-8265-33cbb690f0ed.png)

1. training

(1) 设备检测

```python
#check device
def get_device():
  return 'cuda' if torch.cuda.is_available() else 'cpu'
```

 （2） 固定随机种子(Fix random seeds for reproducibility.)

```python
# fix random seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
```

（3） 设置训练参数

```python
# fix random seed for reproducibility
same_seeds(0) # 固定随机种子是0

# get device 
device = get_device() # 检测是否是GPU
print(f'DEVICE: {device}')  # DEVICE: cuda

# training parameters
num_epoch = 20               # number of training epoch
learning_rate = 0.0001       # learning rate

# the path where checkpoint saved
model_path = './model.ckpt'

# create model, define a loss function, and optimizer
model = Classifier().to(device)
criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) # 如果没有优化器的话，
```

- `Classifier().to(device)`  刚刚检测到GPU, 这时候，这句话的意思是指定模型Classifier() 在GPU上训练
- `criterion = nn.CrossEntropyLoss()`   loss function 用的cross_entropy
- `optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)`  优化器用的是Adam。 优化相关的类定义为 `torch.optim.Optimizer(params, defaults)`  ，其中的两个参数： ）（1） **params (*iterable*) – an iterable of `[torch.Tensor](https://pytorch.org/docs/stable/tensors.html#torch.Tensor)` s or `[dict](https://docs.python.org/3/library/stdtypes.html#dict)` s. Specifies what Tensors should be optimized.（2） defaults – (dict): a dict containing default values of optimization options (used when a parameter group doesn’t specify them).**
    
    因此在这里，我们通过Adam 优化器来优化Model.parameters，learning rate 那个参数很好理解，但是model.parameters是什么呢？我们可以加这段code来test
    

```python
for name, param in model.named_parameters():
    if param.requires_grad:
        print (name, param.data)
```

我们得到

```python
layer1.weight tensor([[-0.0004,  0.0259, -0.0397,  ..., -0.0086, -0.0359, -0.0206],
        ...,
        [-0.0336, -0.0472,  0.0069,  ...,  0.0285,  0.0010, -0.0267]],
       device='cuda:0')
layer1.bias tensor([ 0.0216, -0.0072, -0.0335,  ...,  0.0166,  0.0206, -0.0222],
       device='cuda:0')
layer2.weight tensor([[ 0.0007, -0.0175, -0.0245,  ...,  0.0186, -0.0207, -0.0219],
        ...,
        [-0.0051,  0.0174, -0.0175,  ...,  0.0276,  0.0066, -0.0063]],
       device='cuda:0')
layer2.bias tensor([-1.7310e-02,  2.6931e-02, -2.9340e-02,  2.1919e-03, -2.1392e-02,
        ...,
        -1.4037e-02, -2.4948e-02], device='cuda:0')
layer3.weight tensor([[-0.0229,  0.0165, -0.0294,  ..., -0.0024, -0.0220,  0.0274],
       ...,
        [ 0.0197, -0.0305, -0.0095,  ..., -0.0440,  0.0106,  0.0405]],
       device='cuda:0')
layer3.bias tensor([-0.0040,  0.0222, -0.0349,  0.0126,  0.0334, -0.0176, -0.0330,  0.0335,
       ...,
         0.0425, -0.0355,  0.0008,  0.0205, -0.0400, -0.0027, -0.0057,  0.0366],
       device='cuda:0')
out.weight tensor([[ 0.0632, -0.0372, -0.0169,  ...,  0.0252, -0.0855,  0.0132],
       ...,
        [-0.0784,  0.0149,  0.0422,  ..., -0.0055,  0.0156, -0.0016]],
       device='cuda:0')
out.bias tensor([-0.0201, -0.0187, -0.0713,  0.0180, -0.0648,  0.0731, -0.0351,  0.0582,
       ...,
         0.0020,  0.0353,  0.0455, -0.0360,  0.0414, -0.0442,  0.0576],
       device='cuda:0')
```

(4) 开始训练

```python
# start training

best_acc = 0.0 #
for epoch in range(num_epoch): #num_epoch = 20
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0

    # training
    model.train() # set the model to training mode
    for i, data in enumerate(train_loader): # the shufflng training data
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad() 
        outputs = model(inputs) 
        batch_loss = criterion(outputs, labels)
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        batch_loss.backward()  # Backward pass
        optimizer.step()  # Update parameters

        train_acc += (train_pred.cpu() == labels.cpu()).sum().item()
        train_loss += batch_loss.item()

    # validation
    if len(val_set) > 0:
        model.eval() # set the model to evaluation mode
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                batch_loss = criterion(outputs, labels) 
                _, val_pred = torch.max(outputs, 1) 
            
                val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
                val_loss += batch_loss.item()

            print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f} | Val Acc: {:3.6f} loss: {:3.6f}'.format(
                epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader), val_acc/len(val_set), val_loss/len(val_loader)
            ))

            # if the model improves, save a checkpoint at this epoch
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), model_path)
                print('saving model with acc {:.3f}'.format(best_acc/len(val_set)))
    else:
        print('[{:03d}/{:03d}] Train Acc: {:3.6f} Loss: {:3.6f}'.format(
            epoch + 1, num_epoch, train_acc/len(train_set), train_loss/len(train_loader)
        ))

# if not validating, save the last epoch
if len(val_set) == 0:
    torch.save(model.state_dict(), model_path)
    print('saving model at last epoch')
```

- `inputs, labels = data`  : 从数据加载模块中，`train_set = TIMITDataset(train_x, train_y)` 和`train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)` ，得到input数据和label标签。
- `inputs, labels = inputs.to(device), labels.to(device)` ：指定input 和label在GPU上运行。
- `optimizer.zero_grad()` : 每次变量在back propagated ，也就是`backward()`，梯度会累计而不是替换。在每个batch中，梯度是各不同的 ，所以需要置零。
    
    [why need zero_grad](https://www.notion.so/why-need-zero_grad-e93afae148ea41c2a391a605af3efe3c)
    
- `outputs = model(inputs)`  ,用classifier实现
- `batch_loss = criterion(outputs, labels)`  得到loss function 的值
- `_, train_pred = torch.max(outputs, 1)`  : 每行找到最大的概率，输出
- `batch_loss.backward()`   back forward
1. testing

这个模块和前面训练模块一致，只不过label = None

(1) 创建一个测试数据

```python
# create testing dataset
test_set = TIMITDataset(test, None)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False)

# create model and load weights from checkpoint
model = Classifier().to(device)
model.load_state_dict(torch.load(model_path))
```

(2) 预测

```python
predict = []
model.eval() # set the model to evaluation mode
with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)
```

(3) 

```python
with open('prediction.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))
```
