###############################################################################
# MIT License
#
# Copyright (c) 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2018
# Date Adapted: 2020-11-09
###############################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
from datetime import datetime
import copy

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# datasets
import datasets

# models
from bi_lstm import biLSTM
from lstm import LSTM
from gru import GRU
from peep_lstm import peepLSTM

import numpy as np
import matplotlib.pyplot as plt

# You may want to look into tensorboardX for logging
# from tensorboardX import SummaryWriter

###############################################################################

def train(config, seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Initialize the device which to run the model on
    device = torch.device(config.device)
    print(device)

    # Load dataset
    if config.dataset == 'randomcomb':
        print('Load random combinations dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        dataset = datasets.RandomCombinationsDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

    elif config.dataset == 'bss':
        print('Load bss dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = 2
        config.input_dim = 3
        dataset = datasets.BaumSweetSequenceDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = 4 * config.input_length

    elif config.dataset == 'bipalindrome':
        print('Load binary palindrome dataset ...')
        # Initialize the dataset and data loader
        config.num_classes = config.input_length
        dataset = datasets.BinaryPalindromeDataset(config.input_length)
        data_loader = DataLoader(dataset, config.batch_size, num_workers=1,
                                 drop_last=True)

        config.input_length = config.input_length*4+2-1



    # Setup the model that we are going to use
    if config.model_type == 'LSTM':
        print("Initializing LSTM model ...")
        model = LSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'biLSTM':
        print("Initializing bidirectional LSTM model...")
        model = biLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'GRU':
        print("Initializing GRU model ...")
        model = GRU(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    elif config.model_type == 'peepLSTM':
        print("Initializing peephole LSTM model ...")
        model = peepLSTM(
            config.input_length, config.input_dim,
            config.num_hidden, config.num_classes,
            config.batch_size, device
        ).to(device)

    # Setup the loss and optimizer
    loss_function = torch.nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_list_tmp, accuracy_list_tmp = [], []
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()
        # Move to GPU
        batch_inputs = batch_inputs.to(device)     # [batch_size, seq_length,1]
        batch_targets = batch_targets.to(device)   # [batch_size]
        # Reset for next iteration
        model.zero_grad()

        # Forward pass
        log_probs = model(batch_inputs)
        # Compute the loss, gradients and update network parameters
        loss = loss_function(log_probs, batch_targets)
        loss.backward()

        #######################################################################
        # Check for yourself: what happens here and why?
        #######################################################################
        torch.nn.utils.clip_grad_norm_(model.parameters(),
                                       max_norm=config.max_norm)
        #######################################################################

        optimizer.step()

        predictions = torch.argmax(log_probs, dim=1)
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / log_probs.size(0)

        # print(predictions[0, ...], batch_targets[0, ...])

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if step % 60 == 0:
            loss_list_tmp.append(loss.item())
            accuracy_list_tmp.append(accuracy)
            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                   Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        # Check if training is finished
        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report
            # https://github.com/pytorch/pytorch/pull/9655
            break
        
    print('Done training.')
    loss_list.append(loss_list_tmp)
    accuracy_list.append(accuracy_list_tmp)
    ###########################################################################
    ###########################################################################


if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # dataset
    parser.add_argument('--dataset', type=str, default='randomcomb',
                        choices=['randomcomb', 'bss', 'bipalindrome'],
                        help='Dataset to be trained on.')
    # Model params
    parser.add_argument('--model_type', type=str, default='biLSTM',
                        choices=['LSTM', 'biLSTM', 'GRU', 'peepLSTM'],
                        help='Model type: LSTM, biLSTM, GRU or peepLSTM')
    parser.add_argument('--input_length', type=int, default=6,
                        help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1,
                        help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=1,
                        help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=256,
                        help='Number of hidden units in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=3000,
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    # Misc params
    parser.add_argument('--device', type=str, default="cuda:0",
                        help="Training device 'cpu' or 'cuda:0'")
    parser.add_argument('--gpu_mem_frac', type=float, default=0.5,
                        help='Fraction of GPU memory to allocate')
    parser.add_argument('--log_device_placement', type=bool, default=False,
                        help='Log device placement for debugging')
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')

    config = parser.parse_args()

    # Train the model
    loss_list, accuracy_list = [], []
    train(copy.deepcopy(config), seed=2019)
    train(copy.deepcopy(config), seed=2020)
    train(copy.deepcopy(config), seed=2021)
    #plot loss & accuracy
    loss_list, accuracy_list = np.array(loss_list), np.array(accuracy_list)
    loss_std, accuracy_std = 3*np.std(loss_list, axis=0), 3*np.std(accuracy_list, axis=0)
    loss_avg, accuracy_avg = np.average(loss_list, axis=0), np.average(accuracy_list, axis=0)
    plotRange = 60*np.arange(loss_avg.shape[0])
    plt.plot(plotRange, loss_avg, label='Loss'), plt.fill_between(plotRange, loss_avg - loss_std, loss_avg + loss_std, alpha=0.5) 
    plt.plot(plotRange, accuracy_avg, label='Accuracy'), plt.fill_between(plotRange, accuracy_avg - accuracy_std, accuracy_avg + accuracy_std, alpha=0.5) 
    plt.xlabel('Training Step')
    plt.grid(),plt.legend()
    plt.savefig("%s/%s_%s.png" % (config.summary_path,config.model_type, config.dataset), bbox_inches='tight')
    plt.show()

#python train.py --dataset bss --model_type LSTM --train_steps 1000 --learning_rate 0.0001 --batch_size 256 --num_hidden 256 --device cpu --input_length 4
#python train.py --dataset bss --model_type GRU --train_steps 1000 --learning_rate 0.0001 --batch_size 256 --num_hidden 256 --device cpu --input_length 4

