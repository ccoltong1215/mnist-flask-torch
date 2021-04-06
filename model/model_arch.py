import numpy as np

import torch
import torch.nn as nn
from collections import OrderedDict

class LeNet5(nn.Module):
    """ LeNet-5 (LeCun et al., 1998)

        - For a detailed architecture, refer to the lecture note
        - Freely choose activation functions as you want
        - For subsampling, use max pooling with kernel_size = (2,2)
        - Output should be a logit vector
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2, bias=True)),
            ('relu1', nn.ReLU()),
            ('max_pool_1', nn.MaxPool2d(kernel_size=2, stride=2)),
            ('conv2', nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1, padding=0, bias=True)),
            ('relu2', nn.ReLU()),
            ('max_pool_2', nn.MaxPool2d(kernel_size=2, stride=2))
        ]))
        self.head = nn.Sequential(OrderedDict([
            ('fc1', nn.Linear(16 * 5 * 5,
                              120)),
            # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
            ('relu3', nn.ReLU()),
            ('fc2', nn.Linear(120, 84)),  # convert matrix with 120 features to a matrix of 84 features (columns)
            ('relu4', nn.ReLU()),
            ('fc3', nn.Linear(84, 10))
        ]))

    def forward(self, img):

        img = self.body(img)
        img = img.view(-1, 16 * 5 * 5)
        output = self.head(img)

        return output
class CustomMLP(nn.Module):
    """ Your custom MLP model

        - Note that the number of model parameters should be about the same
          with LeNet-5
    """

        # write your codes here
    def __init__(self):
        super(CustomMLP, self).__init__()
        # self.cnn= nn.Conv2d(in_channels=1, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

        self.body = nn.Sequential(OrderedDict([
            ('conv1', nn.Linear(28*28*1,28*28*6)),
            ('relu1', nn.ReLU()),
            ('max_pool_1', nn.Linear(28*28*6,14*14*6)),
            ('conv2', nn.Linear(14*14*6,10*10*16)),
            ('relu2', nn.ReLU()),
            ('max_pool_2', nn.Linear(10*10*16,5*5*16))
        ]))

        self.head=nn.Sequential(OrderedDict([
            ('fc1' , nn.Linear(5*5*16,120
                                 )),  # convert matrix with 16*5*5 (= 400) features to a matrix of 120 features (columns)
            ('relu3', nn.ReLU()),
            ('fc2' , nn.Linear(120, 84)),  # convert matrix with 120 features to a matrix of 84 features (columns)
            ('relu4', nn.ReLU()),
            ('fc3' , nn.Linear(84, 10))
        ]))


    def forward(self, img):
        # img = self.cnn(img)
        img = img.view(-1, 28*28*1)
        img = self.body(img)
        output = self.head(img)

        return output