"""
This module implements a multi-layer perceptron (MLP) in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch
import numpy as np


class MLP(nn.Module):
    """
    This class implements a Multi-layer Perceptron in PyTorch.
    It handles the different layers and parameters of the model.
    Once initialized an MLP object can perform forward.
    """
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes MLP object.
        
        Args:
          n_inputs: number of inputs.
          n_hidden: list of ints, specifies the number of units
                    in each linear layer. If the list is empty, the MLP
                    will not have any linear layers, and the model
                    will simply perform a multinomial logistic regression.
          n_classes: number of classes of the classification problem.
                     This number is required in order to specify the
                     output dimensions of the MLP
    
        TODO:
        Implement initialization of the network.
        """
        super(MLP, self).__init__()
        layers = nn.ModuleList()
        if n_classes == 0:
          layers.append(nn.Linear(n_inputs, n_classes))    # first and only layer
        else:
          layers.append(nn.Linear(n_inputs, n_hidden[0]))  # first layer
        layers.append(nn.ELU())

        for index, layerInputs in enumerate(n_hidden):
          if index == len(n_hidden) - 1:
            layers.append(nn.Linear(layerInputs, n_classes))             # last layer
          else:
            layers.append(nn.Linear(layerInputs, n_hidden[index+1]))  # hidden layer
          layers.append(nn.ELU())
        
        layers.append(nn.Softmax(dim=1))
        self.layers = layers
    
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
        
        out = x
        for layer in self.layers:
          out = layer(out)
        
        return out
