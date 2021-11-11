# Homework 2-1 Phoneme Classification 

* Slides: https://speech.ee.ntu.edu.tw/~hylee/ml/ml2021-course-data/hw/HW02/HW02.pdf
* Video (Chinese): https://youtu.be/PdjXnQbu2zo
* Video (English): https://youtu.be/ESRr-VCykBs

## Object
Solve a Multiclass Classification problem(Framewise phoneme prediction from speech) with DNN

## Description & task
<img width="1174" alt="未命名" src="https://user-images.githubusercontent.com/69283174/141235528-a7e3daff-b330-42fe-9d74-3cc0a765ed6a.png">

## Data

### Data preprocessing
<img width="1064" alt="2" src="https://user-images.githubusercontent.com/69283174/141235715-255c322c-276f-44c0-976b-8d6c44430f96.png">

可以将每25ms的声音转化为一个向量，最近使用的是通过filter bank output将其转化为一个维度为80的向量。
生成下一个向量的时候，不是直接向后移动25，而是小于25的10，因此生成的向量是包含重复内容的。


<img width="1207" alt="3" src="https://user-images.githubusercontent.com/69283174/141269582-90bd132d-3d23-4c56-b561-c9d734f5a1fa.png">

### Data analysis
<img width="379" alt="4" src="https://user-images.githubusercontent.com/69283174/141303343-c3ac4f71-8571-47b5-8f05-3a2e2d3ca557.png">
     
Size of training data: (1229932, 429)

Size of testing data: (451552, 429)

input: 11 frames 的处理过的特征，包括前五frames+ 当前frame +后五frames。每一帧是39维，所以input 11✖️39=429维度。图中的label指的 就是中间那帧的特征。

## Hint
### Simple baseline 
You should able to pass the simple baseline using the sample code provided.
### Strong baseline
Model architecture (layers? dimension? activation function?)
Training (batch size? optimizer? learning rate? epoch?)
Tips (batch norm? dropout? regularization?)
 
