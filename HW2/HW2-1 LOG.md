# HW2-1 LOG

(1) when learning rate is not enough small,  no matter what change cannot improve the bad accuracy 

# E1: (test sample)

VAL_RATIO = 0.2

BATCH_SIZE = 64

```jsx
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 39) 

        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
        
        return x
```

num_epoch = 20               # number of training epoch

learning_rate = 0.01       # learning rate

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## output:

```jsx
[001/020] Train Acc: 0.385513 Loss: 2.222536 | Val Acc: 0.315183 loss: 2.503171
saving model with acc 0.315
[002/020] Train Acc: 0.296476 Loss: 2.559029 | Val Acc: 0.215820 loss: 2.654881
[003/020] Train Acc: 0.229326 Loss: 2.725262 | Val Acc: 0.172952 loss: 2.888282
[004/020] Train Acc: 0.178735 Loss: 2.884237 | Val Acc: 0.189400 loss: 2.878119
[005/020] Train Acc: 0.183065 Loss: 2.897741 | Val Acc: 0.182270 loss: 2.887694
[006/020] Train Acc: 0.180497 Loss: 2.872780 | Val Acc: 0.181347 loss: 2.886141
[007/020] Train Acc: 0.180099 Loss: 2.889818 | Val Acc: 0.190189 loss: 2.897130
[008/020] Train Acc: 0.186835 Loss: 2.994896 | Val Acc: 0.179229 loss: 2.885835
[009/020] Train Acc: 0.176087 Loss: 2.894241 | Val Acc: 0.179721 loss: 2.898038
[010/020] Train Acc: 0.182465 Loss: 2.879075 | Val Acc: 0.185095 loss: 2.899888
[011/020] Train Acc: 0.190957 Loss: 2.884465 | Val Acc: 0.193230 loss: 2.893866
[012/020] Train Acc: 0.191216 Loss: 2.896750 | Val Acc: 0.192226 loss: 2.904715
[013/020] Train Acc: 0.190678 Loss: 2.899383 | Val Acc: 0.192226 loss: 2.903369
[014/020] Train Acc: 0.191258 Loss: 2.899205 | Val Acc: 0.192226 loss: 2.903468
[015/020] Train Acc: 0.190793 Loss: 2.899320 | Val Acc: 0.192226 loss: 2.904766
[016/020] Train Acc: 0.190891 Loss: 2.899328 | Val Acc: 0.189937 loss: 2.904458
[017/020] Train Acc: 0.191099 Loss: 2.899262 | Val Acc: 0.189937 loss: 2.903503
[018/020] Train Acc: 0.191162 Loss: 2.899147 | Val Acc: 0.189937 loss: 2.904390
[019/020] Train Acc: 0.191002 Loss: 2.899346 | Val Acc: 0.192226 loss: 2.904639
[020/020] Train Acc: 0.190887 Loss: 2.899287 | Val Acc: 0.189937 loss: 2.903299
```

# E2:  (test learning rate)

VAL_RATIO = 0.2

BATCH_SIZE = 64

```jsx
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 1024)
        self.layer2 = nn.Linear(1024, 512)
        self.layer3 = nn.Linear(512, 128)
        self.out = nn.Linear(128, 39)

        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
        
        return x
```

num_epoch = 20               # number of training epoch

learning_rate = 0.01       # learning rate

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## output:

learning_rate = 0.01       # learning rate

```jsx
[001/020] Train Acc: 0.301442 Loss: 2.486272 | Val Acc: 0.212658 loss: 2.782219
saving model with acc 0.213
[002/020] Train Acc: 0.172410 Loss: 2.931863 | Val Acc: 0.142877 loss: 3.064177
[003/020] Train Acc: 0.146519 Loss: 3.019358 | Val Acc: 0.143292 loss: 2.984652
[004/020] Train Acc: 0.155213 Loss: 2.968299 | Val Acc: 0.154045 loss: 2.970131
[005/020] Train Acc: 0.153951 Loss: 2.962578 | Val Acc: 0.166968 loss: 2.939145
[006/020] Train Acc: 0.164781 Loss: 2.931896 | Val Acc: 0.164318 loss: 2.943726
[007/020] Train Acc: 0.164349 Loss: 2.932337 | Val Acc: 0.164318 loss: 2.943495
[008/020] Train Acc: 0.164459 Loss: 2.932158 | Val Acc: 0.164318 loss: 2.942464
[009/020] Train Acc: 0.164509 Loss: 2.932214 | Val Acc: 0.164318 loss: 2.943729
[010/020] Train Acc: 0.164330 Loss: 2.932188 | Val Acc: 0.164318 loss: 2.942202
[011/020] Train Acc: 0.164485 Loss: 2.932203 | Val Acc: 0.164318 loss: 2.942838
[012/020] Train Acc: 0.164355 Loss: 2.932201 | Val Acc: 0.164318 loss: 2.943290
[013/020] Train Acc: 0.164481 Loss: 2.932281 | Val Acc: 0.164318 loss: 2.943659
[014/020] Train Acc: 0.164299 Loss: 2.932225 | Val Acc: 0.155870 loss: 2.942429
[015/020] Train Acc: 0.164397 Loss: 2.932249 | Val Acc: 0.164318 loss: 2.942855
[016/020] Train Acc: 0.164517 Loss: 2.932219 | Val Acc: 0.164318 loss: 2.943058
[017/020] Train Acc: 0.164570 Loss: 2.932252 | Val Acc: 0.164318 loss: 2.943089
[018/020] Train Acc: 0.164358 Loss: 2.932162 | Val Acc: 0.164318 loss: 2.943179
[019/020] Train Acc: 0.164533 Loss: 2.932203 | Val Acc: 0.164318 loss: 2.942513
[020/020] Train Acc: 0.164472 Loss: 2.932263 | Val Acc: 0.164318 loss: 2.943365
```

learning_rate = 0.001       # learning rate (其他不变）

```jsx
[001/020] Train Acc: 0.639094 Loss: 1.153346 | Val Acc: 0.673735 loss: 1.051041
saving model with acc 0.674
[002/020] Train Acc: 0.692916 Loss: 0.968140 | Val Acc: 0.682565 loss: 1.014027
saving model with acc 0.683
[003/020] Train Acc: 0.712380 Loss: 0.901045 | Val Acc: 0.687252 loss: 1.013957
saving model with acc 0.687
[004/020] Train Acc: 0.727031 Loss: 0.855035 | Val Acc: 0.695622 loss: 0.995581
saving model with acc 0.696
[005/020] Train Acc: 0.737927 Loss: 0.818477 | Val Acc: 0.691049 loss: 1.038447
[006/020] Train Acc: 0.747656 Loss: 0.788787 | Val Acc: 0.690646 loss: 1.050367
[007/020] Train Acc: 0.755341 Loss: 0.762790 | Val Acc: 0.694772 loss: 1.049173
[008/020] Train Acc: 0.762455 Loss: 0.740913 | Val Acc: 0.694451 loss: 1.083187
[009/020] Train Acc: 0.769253 Loss: 0.721047 | Val Acc: 0.693890 loss: 1.093981
[010/020] Train Acc: 0.775271 Loss: 0.702906 | Val Acc: 0.694533 loss: 1.095010
[011/020] Train Acc: 0.781271 Loss: 0.685831 | Val Acc: 0.689618 loss: 1.119709
[012/020] Train Acc: 0.785991 Loss: 0.670370 | Val Acc: 0.694464 loss: 1.129954
[013/020] Train Acc: 0.790226 Loss: 0.657696 | Val Acc: 0.690602 loss: 1.178663
[014/020] Train Acc: 0.795301 Loss: 0.644008 | Val Acc: 0.690272 loss: 1.187856
[015/020] Train Acc: 0.799075 Loss: 0.631475 | Val Acc: 0.693516 loss: 1.167782
[016/020] Train Acc: 0.803304 Loss: 0.619142 | Val Acc: 0.690524 loss: 1.227403
[017/020] Train Acc: 0.807020 Loss: 0.607034 | Val Acc: 0.691614 loss: 1.233623
[018/020] Train Acc: 0.810951 Loss: 0.595487 | Val Acc: 0.689569 loss: 1.269074
[019/020] Train Acc: 0.814019 Loss: 0.587969 | Val Acc: 0.684841 loss: 1.281010
[020/020] Train Acc: 0.816770 Loss: 0.578792 | Val Acc: 0.688984 loss: 1.318732
```

learning_rate = 0.0001       # learning rate (其他不变）

```jsx
[001/020] Train Acc: 0.620783 Loss: 1.226873 | Val Acc: 0.677768 loss: 1.020711
saving model with acc 0.678
[002/020] Train Acc: 0.691674 Loss: 0.964206 | Val Acc: 0.694789 loss: 0.948021
saving model with acc 0.695
[003/020] Train Acc: 0.717485 Loss: 0.869706 | Val Acc: 0.704830 loss: 0.911025
saving model with acc 0.705
[004/020] Train Acc: 0.736914 Loss: 0.802856 | Val Acc: 0.709964 loss: 0.895477
saving model with acc 0.710
[005/020] Train Acc: 0.753045 Loss: 0.748859 | Val Acc: 0.713542 loss: 0.883774
saving model with acc 0.714
[006/020] Train Acc: 0.766305 Loss: 0.702210 | Val Acc: 0.716505 loss: 0.877961
saving model with acc 0.717
[007/020] Train Acc: 0.779085 Loss: 0.660410 | Val Acc: 0.713542 loss: 0.900045
[008/020] Train Acc: 0.790780 Loss: 0.622188 | Val Acc: 0.713497 loss: 0.904404
[009/020] Train Acc: 0.801800 Loss: 0.586467 | Val Acc: 0.709310 loss: 0.926278
[010/020] Train Acc: 0.811833 Loss: 0.554274 | Val Acc: 0.708729 loss: 0.942616
[011/020] Train Acc: 0.822031 Loss: 0.523190 | Val Acc: 0.706163 loss: 0.962550
[012/020] Train Acc: 0.830516 Loss: 0.494695 | Val Acc: 0.708285 loss: 1.000067
[013/020] Train Acc: 0.839623 Loss: 0.467466 | Val Acc: 0.703066 loss: 1.025749
[014/020] Train Acc: 0.847337 Loss: 0.442944 | Val Acc: 0.704033 loss: 1.043199
[015/020] Train Acc: 0.855354 Loss: 0.418980 | Val Acc: 0.700687 loss: 1.087938
[016/020] Train Acc: 0.862199 Loss: 0.397589 | Val Acc: 0.697094 loss: 1.127987
[017/020] Train Acc: 0.868697 Loss: 0.377235 | Val Acc: 0.695240 loss: 1.162218
[018/020] Train Acc: 0.874887 Loss: 0.358407 | Val Acc: 0.692663 loss: 1.206702
[019/020] Train Acc: 0.881027 Loss: 0.341017 | Val Acc: 0.691890 loss: 1.236701
[020/020] Train Acc: 0.886043 Loss: 0.323901 | Val Acc: 0.691211 loss: 1.264787
```

learning_rate = 0.000001       # learning rate (其他不变）

```jsx
[001/020] Train Acc: 0.268654 Loss: 2.836264 | Val Acc: 0.382910 loss: 2.286973
saving model with acc 0.383
[002/020] Train Acc: 0.412471 Loss: 2.095618 | Val Acc: 0.453991 loss: 1.908675
saving model with acc 0.454
[003/020] Train Acc: 0.464989 Loss: 1.846138 | Val Acc: 0.490896 loss: 1.739259
saving model with acc 0.491
[004/020] Train Acc: 0.492200 Loss: 1.720606 | Val Acc: 0.514332 loss: 1.641316
saving model with acc 0.514
[005/020] Train Acc: 0.511103 Loss: 1.641422 | Val Acc: 0.531333 loss: 1.574290
saving model with acc 0.531
[006/020] Train Acc: 0.526509 Loss: 1.583376 | Val Acc: 0.545061 loss: 1.522322
saving model with acc 0.545
[007/020] Train Acc: 0.538914 Loss: 1.537582 | Val Acc: 0.556899 loss: 1.480678
saving model with acc 0.557
[008/020] Train Acc: 0.549157 Loss: 1.499697 | Val Acc: 0.565762 loss: 1.445103
saving model with acc 0.566
[009/020] Train Acc: 0.557590 Loss: 1.467654 | Val Acc: 0.574323 loss: 1.415894
saving model with acc 0.574
[010/020] Train Acc: 0.565109 Loss: 1.439888 | Val Acc: 0.581246 loss: 1.389335
saving model with acc 0.581
[011/020] Train Acc: 0.571641 Loss: 1.415596 | Val Acc: 0.586466 loss: 1.367368
saving model with acc 0.586
[012/020] Train Acc: 0.577555 Loss: 1.393895 | Val Acc: 0.592137 loss: 1.346962
saving model with acc 0.592
[013/020] Train Acc: 0.582649 Loss: 1.374376 | Val Acc: 0.597166 loss: 1.328500
saving model with acc 0.597
[014/020] Train Acc: 0.587531 Loss: 1.356747 | Val Acc: 0.601430 loss: 1.312477
saving model with acc 0.601
[015/020] Train Acc: 0.591820 Loss: 1.340527 | Val Acc: 0.605833 loss: 1.296849
saving model with acc 0.606
[016/020] Train Acc: 0.595696 Loss: 1.325579 | Val Acc: 0.609134 loss: 1.283659
saving model with acc 0.609
[017/020] Train Acc: 0.599560 Loss: 1.311799 | Val Acc: 0.612545 loss: 1.271402
saving model with acc 0.613
[018/020] Train Acc: 0.602817 Loss: 1.298921 | Val Acc: 0.616163 loss: 1.258657
saving model with acc 0.616
[019/020] Train Acc: 0.606249 Loss: 1.286851 | Val Acc: 0.619195 loss: 1.248148
saving model with acc 0.619
[020/020] Train Acc: 0.609111 Loss: 1.275519 | Val Acc: 0.621163 loss: 1.237663
saving model with acc 0.621
```

<aside>
💡 val acc 没有提升太多，但是train 结果好的特别好，初步判断是过拟合

</aside>

# E3: (simplify the model)

VAL_RATIO = 0.2

BATCH_SIZE = 64

```jsx
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.layer1 = nn.Linear(429, 512)
        self.layer2 = nn.Linear(512, 256)
        self.layer3 = nn.Linear(256, 128)
        self.out = nn.Linear(128, 39)

        self.act_fn = nn.ReLU()

    def forward(self, x):
        x = self.layer1(x)
        x = self.act_fn(x)

        x = self.layer2(x)
        x = self.act_fn(x)

        x = self.layer3(x)
        x = self.act_fn(x)

        x = self.out(x)
        
        return x
```

num_epoch = 20               # number of training epoch

learning_rate = 0.0001     # learning rate

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

## OUTPUT:

```jsx
[001/020] Train Acc: 0.602569 Loss: 1.294777 | Val Acc: 0.663572 loss: 1.072878
saving model with acc 0.664
[002/020] Train Acc: 0.671578 Loss: 1.036310 | Val Acc: 0.680520 loss: 0.999899
saving model with acc 0.681
[003/020] Train Acc: 0.694313 Loss: 0.953446 | Val Acc: 0.694126 loss: 0.953026
saving model with acc 0.694
[004/020] Train Acc: 0.708864 Loss: 0.899166 | Val Acc: 0.701627 loss: 0.922313
saving model with acc 0.702
[005/020] Train Acc: 0.720969 Loss: 0.857733 | Val Acc: 0.703964 loss: 0.911738
saving model with acc 0.704
[006/020] Train Acc: 0.730234 Loss: 0.824986 | Val Acc: 0.706371 loss: 0.907786
saving model with acc 0.706
[007/020] Train Acc: 0.738401 Loss: 0.796515 | Val Acc: 0.705151 loss: 0.909180
[008/020] Train Acc: 0.745227 Loss: 0.772834 | Val Acc: 0.708696 loss: 0.896669
saving model with acc 0.709
[009/020] Train Acc: 0.751408 Loss: 0.751057 | Val Acc: 0.709013 loss: 0.901347
saving model with acc 0.709
[010/020] Train Acc: 0.757186 Loss: 0.731437 | Val Acc: 0.707086 loss: 0.907378
[011/020] Train Acc: 0.762303 Loss: 0.713557 | Val Acc: 0.709920 loss: 0.903052
saving model with acc 0.710
[012/020] Train Acc: 0.767126 Loss: 0.697839 | Val Acc: 0.707928 loss: 0.911117
[013/020] Train Acc: 0.771747 Loss: 0.682584 | Val Acc: 0.709245 loss: 0.910126
[014/020] Train Acc: 0.775892 Loss: 0.668455 | Val Acc: 0.708810 loss: 0.915293
[015/020] Train Acc: 0.779428 Loss: 0.655429 | Val Acc: 0.708184 loss: 0.918671
[016/020] Train Acc: 0.784112 Loss: 0.642667 | Val Acc: 0.709298 loss: 0.923990
[017/020] Train Acc: 0.786850 Loss: 0.631287 | Val Acc: 0.707814 loss: 0.932612
[018/020] Train Acc: 0.790844 Loss: 0.620293 | Val Acc: 0.705147 loss: 0.943557
[019/020] Train Acc: 0.793815 Loss: 0.609675 | Val Acc: 0.705940 loss: 0.953038
[020/020] Train Acc: 0.796960 Loss: 0.599569 | Val Acc: 0.704867 loss: 0.962617
```
