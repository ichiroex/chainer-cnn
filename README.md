# chainer-cnn
Simple Convolutional Neural Network for sentence classification (positive or negative) .  

# Requirements
This code is written in Python with Chainer which is framework of Deep Neural Network.  
Please download `GoogleNews-vectors-negative300.bin.gz` from [this site](https://code.google.com/archive/p/word2vec/) and put it in the same directory as these codes.  

# Usage
```
  $ python train_cnn.py [--gpu 1 or 0]   
```

# Optional arguments
```
  -h, --help            show this help message and exit
  --gpu   GPU           1: use gpu, 0: use cpu
  --data  DATA          an input data file
  --epoch EPOCH         number of epochs to learn
  --batchsize BATCHSIZE
                        learning minibatch size
  --nunits NUNITS       number of units
```

# Data format for input data
  - [0 or 1] [Sequence of words]  
    - 1 and 0 are positive and negative, respectively.  

## Examples
```
1 That was so beautiful that it can't be put into words . (POSITIVE SETENCE)
0 I do not want to go to school because I do like to study math . (NEGATIVE SENTENCE)
```
