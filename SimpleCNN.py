# -*- coding: utf-8 -*-

import chainer
from chainer import cuda, Function, gradient_check, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class SimpleCNN(Chain):
    
    def __init__(self, input_channel, output_channel, filter_height, filter_width, mid_units, n_units, n_label):
        super(SimpleCNN, self).__init__(
            conv1 = L.Convolution2D(input_channel, output_channel, (filter_height, filter_width)),
            l1    = L.Linear(mid_units, n_units),
            l2    = L.Linear(n_units,  n_label),
        )
    
    #Classifier によって呼ばれる
    def __call__(self, x):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3)
        h2 = F.dropout(F.relu(self.l1(h1)))
        y = self.l2(h2)
        return y

    def forward(self, x, t, train=True):
        h1 = F.max_pooling_2d(F.relu(self.conv1(x)), 3)
        h2 = F.dropout(F.relu(self.l1(h1)), train=train)
        y = self.l2(h2)
        return F.softmax_cross_entropy(y, t), F.accuracy(y, t)
