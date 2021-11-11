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

首先，输入的声音信号是一串的向量。可以将每25ms的声音转化为一个向量，通过MFCC，将其转化为一个维度为39的维度。
生成一下一个向量的时候，往后移动10ms，这样生成的向量之间是包含重复内容的。


