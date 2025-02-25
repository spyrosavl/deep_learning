# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn
import torch

class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):
        super(TextGenerationModel, self).__init__()
        self.device = device
        self.lstm_num_hidden, self.batch_size, self.lstm_num_layers = lstm_num_hidden, batch_size, lstm_num_layers
        self.embeddings = nn.Embedding(num_embeddings=vocabulary_size, embedding_dim=lstm_num_hidden)
        self.lstm = nn.LSTM(input_size=lstm_num_hidden, hidden_size=lstm_num_hidden,num_layers=lstm_num_layers, batch_first=True)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)
        
        nn.init.kaiming_normal_(self.linear.weight)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
        
        self.to(device)

    def forward(self, input, hiddenState=None):
        features = self.embeddings(input)
        if hiddenState:
            output, hiddenState = self.lstm(features, hiddenState)
        else:
            output, hiddenState = self.lstm(features)
        out = self.linear(output)
        return out, hiddenState
    
    def init_state(self, sequence_length):
        return (torch.zeros(self.lstm_num_layers, sequence_length, self.lstm_num_hidden),
                torch.zeros(self.lstm_num_layers, sequence_length, self.lstm_num_hidden))
