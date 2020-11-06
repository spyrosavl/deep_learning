"""
This module implements a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class PreActResNetBlock(nn.Module):

    def __init__(self, c_in, double=True):
        """
        Inputs:
            c_in - Number of input features
            act_fn - Activation class constructor (e.g. nn.ReLU)
            subsample - If True, we want to apply a stride inside the block and reduce the output shape by 2 in height and width
            c_out - Number of output features. Note that this is only relevant if subsample is True, as otherwise, c_out = c_in
        """
        super().__init__()
        c_out = c_in
        self.double = double
        self.netA = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        )

        self.netB = nn.Sequential(
            nn.BatchNorm2d(c_in),
            nn.ReLU(),
            nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        z = self.netA(x)
        if self.double:
          z = self.netB(z)
        out = z + x
        return out

class ConvNet(nn.Module):
    """
    This class implements a Convolutional Neural Network in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an ConvNet object can perform forward.
    """
    
    def __init__(self, n_channels, n_classes):
        """
        Initializes ConvNet object.
        
        Args:
          n_channels: number of input channels
          n_classes: number of classes of the classification problem
          
        
        TODO:
        Implement initialization of the network.
        """
        super().__init__()
        self.input_net = nn.Sequential(
                nn.Conv2d(n_channels, 64, kernel_size=3, padding=1, bias=False),
                PreActResNetBlock(64, double=False),
            )
        
        blocks = []
        blocks += ([
          nn.Conv2d(64, 128, kernel_size=1, padding=0, bias=False),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          PreActResNetBlock(128, double=True),
        ])
        blocks += ([
          nn.Conv2d(128, 256, kernel_size=1, padding=0, bias=False),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          PreActResNetBlock(256, double=True),
        ])
        blocks += ([
          nn.Conv2d(256, 512, kernel_size=1, padding=0, bias=False),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          PreActResNetBlock(512, double=True),
        ])
        blocks += ([
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
          PreActResNetBlock(512, double=True),
          nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        ])
        self.blocks = nn.Sequential(*blocks)

        self.output_net = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=512, out_features=n_classes)
            )
        # print(self.input_net)
        # print(self.blocks)
        # print(self.output_net)
    
    def forward(self, x):
        """
        Performs forward pass of the input. Here an input tensor x is transformed through
        several layer transformations.
        
        Args:
          x: input to the network
        Returns:
          out: outputs of the network
        
        TODO:
        Implement forward pass of the network.
        """
        x = self.input_net(x)
        x = self.blocks(x)
        x = self.output_net(x)
        return x
  
