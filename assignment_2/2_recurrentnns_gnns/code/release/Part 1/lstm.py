"""
This module implements a LSTM model in PyTorch.
You should fill in code into indicated sections.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, num_classes,
                 batch_size, device):
        super(LSTM, self).__init__()
        print("Seq len: %d, Input dim: %d, No of classes %d" % (seq_length, input_dim, num_classes))
        self.embeds = nn.Embedding(input_dim, input_dim)
        #weight matrix
        self.W_gx = nn.Parameter(torch.randn(hidden_dim, input_dim))
        nn.init.kaiming_normal_(self.W_gx)
        self.W_gh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W_gh)
        self.W_ix = nn.Parameter(torch.randn(hidden_dim, input_dim))
        nn.init.kaiming_normal_(self.W_ix)
        self.W_ih = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W_ih)
        self.W_fx = nn.Parameter(torch.randn(hidden_dim, input_dim))
        nn.init.kaiming_normal_(self.W_fx)
        self.W_fh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W_fh)
        self.W_ox = nn.Parameter(torch.randn(hidden_dim, input_dim))
        nn.init.kaiming_normal_(self.W_ox)
        self.W_oh = nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        nn.init.kaiming_normal_(self.W_oh)
        self.W_ph = nn.Parameter(torch.randn(num_classes, hidden_dim))
        nn.init.kaiming_normal_(self.W_ph)
        
        #bias matrix
        self.b_g = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.b_i = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.b_f = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.b_o = nn.Parameter(torch.zeros(hidden_dim, 1))
        self.b_p = nn.Parameter(torch.zeros(num_classes, 1))

        # activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.log_softmax = nn.LogSoftmax(dim=0)
        self.to(device)

    def forward(self, x):
        h_t = torch.zeros(self.hidden_dim, self.batch_size)
        c_t = torch.zeros(self.hidden_dim, self.batch_size)
        x = x.squeeze()
        x = self.embeds(x.to(torch.int64))
        for i in range(x.shape[1]):
            xi = x[:,i,:].permute(1,0)
            g_t = self.tanh(self.W_gx @ xi + self.W_gh @ h_t + self.b_g)
            i_t = self.sigmoid(self.W_ix @ xi + self.W_ih @ h_t + self.b_i)
            f_t = self.sigmoid(self.W_fx @ xi + self.W_fh @ h_t + self.b_f)
            o_t = self.sigmoid(self.W_ox @ xi+ self.W_oh @ h_t + self.b_o)
            c_t = g_t * i_t + c_t * f_t
            h_t = self.tanh(c_t) * o_t
        
        y_t = self.W_ph @ h_t + self.b_p
        y_t = self.log_softmax(y_t)
        y_t = y_t.permute(1,0)
        return y_t
