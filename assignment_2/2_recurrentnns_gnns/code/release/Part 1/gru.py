"""
This module implements a GRU in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class GRU(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):

        super(GRU, self).__init__()

        self._seq_length = seq_length
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self._num_classes = num_classes
        self._batch_size = batch_size
        self._device = device

        #print("Seq len: %d, Input dim: %d, No of classes %d" % (seq_length, input_dim, num_classes))
        self.embeds = nn.Embedding(input_dim, input_dim)
        #weight matrix
        self.W_z = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        nn.init.kaiming_normal_(self.W_z)
        self.U_z = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.U_z)
        self.W_r = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        nn.init.kaiming_normal_(self.W_r)
        self.U_r = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.U_r)
        self.W = nn.Parameter(torch.zeros(hidden_dim, input_dim))
        nn.init.kaiming_normal_(self.W)
        self.U = nn.Parameter(torch.zeros(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.U)
        self.W_ph = nn.Parameter(torch.zeros(num_classes, hidden_dim))
        nn.init.kaiming_normal_(self.W_ph)
        
        #bias matrix
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

        # activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.log_softmax = nn.LogSoftmax(dim=0)
        self.to(device)

    def forward(self, x):
        h_t = torch.zeros(self._hidden_dim, self._batch_size)
        x = x.squeeze()
        x = self.embeds(x.to(torch.int64))
        for i in range(x.shape[1]):
            xi = x[:,i,:].permute(1,0)
            z_t = self.sigmoid(self.W_z @ xi + self.U_z @ h_t)
            r_t = self.sigmoid(self.W_r @ xi + self.U_r @ h_t)
            h_hat = self.tanh(self.W @ xi + r_t * (self.U @ h_t))
            h_t = self.tanh(z_t * h_t + (1 - z_t) * h_hat)
        
        y_t = self.W_ph @ h_t + self.b_p
        y_t = self.log_softmax(y_t)
        y_t = y_t.permute(1,0)
        return y_t
