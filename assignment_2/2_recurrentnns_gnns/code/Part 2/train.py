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

import os
import time
from datetime import datetime
import argparse
import random

import numpy as np
import math

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import TextDataset
from model import TextGenerationModel

###############################################################################

def predict_with_temperature(config, dataset, model, textOriginal, next_chars=30, temperature=2):
    model.eval()
    text = list(textOriginal)
    hiddenState = model.init_state(1)

    for i in range(0, next_chars):
        x = torch.tensor([[dataset._char_to_ix[c] for c in text[i:]]])
        y_pred, hiddenState = model(x, hiddenState)
        last_char = temperature * y_pred[0][-1]
        p = torch.nn.functional.softmax(last_char, dim=0).detach().numpy()
        word_index = np.random.choice(len(last_char), p=p)
        text.append(dataset._ix_to_char[word_index])

    text = [t for t in text if t != '\n']
    text = ''.join(text)
    return text

def predict_greedy(config, dataset, model, textOriginal, next_chars=30):
    model.eval()
    text = list(textOriginal)
    hiddenState = model.init_state(1)

    for i in range(0, next_chars):
        x = torch.tensor([[dataset._char_to_ix[c] for c in text[i:]]])
        y_pred, hiddenState = model(x, hiddenState)
        last_char = y_pred[0][-1].argmax().item()
        text.append(dataset._ix_to_char[last_char])

    text = [t for t in text if t != '\n']
    text = ''.join(text)
    return text

def train(config):

    # Initialize the device which to run the model on
    device = torch.device(config.device)

    # Initialize the dataset and data loader (note the +1)
    dataset = TextDataset(config.txt_file, config.seq_length)
    data_loader = DataLoader(dataset, config.batch_size)

    # Initialize the model that we are going to use
    model = TextGenerationModel(config.batch_size, config.seq_length, dataset.vocab_size, config.lstm_num_hidden, config.lstm_num_layers, config.device)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    #hiddenState = model.init_state(config.seq_length)
    startTime = time.time()
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Only for time measurement of step through network
        t1 = time.time()
        # Move to GPU
        batch_inputs = torch.stack(batch_inputs).to(device).t() #[batch size, no of chars]
        batch_targets = torch.stack(batch_targets).to(device)
        #train model
        model.train()
        # Reset for next iteration
        model.zero_grad()
        # Forward pass
        log_probs, hiddenState = model(batch_inputs)#, hiddenState) #[batch size, classes, sequence len]
        #hiddenState = (hiddenState[0].detach(), hiddenState[1].detach())

        # Compute the loss, gradients and update network parameters
        log_probs = log_probs.permute(1, 2, 0)
        loss = criterion(log_probs, batch_targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        optimizer.step()
        predictions = torch.argmax(log_probs, dim=1)
        correct = (predictions == batch_targets).sum().item()
        accuracy = correct / (log_probs.size(0) * log_probs.size(2))

        # Just for time measurement
        t2 = time.time()
        examples_per_second = config.batch_size/float(t2-t1)

        if (step + 1) % config.print_every == 0:

            print("[{}] Train Step {:04d}/{:04d}, Batch Size = {}, \
                    Examples/Sec = {:.2f}, "
                  "Accuracy = {:.2f}, Loss = {:.3f}".format(
                    datetime.now().strftime("%Y-%m-%d %H:%M"), step,
                    config.train_steps, config.batch_size, examples_per_second,
                    accuracy, loss
                    ))

        if (step + 1) % config.sample_every == 0:
            startText = 'America is'
            if config.greed_sampling:
                text = predict_with_temperature(config, dataset, model, startText, config.seq_length)
            else:
                text = predict_with_temperature(config, dataset, model, startText, config.seq_length)
            exportFile = '%stextGenerated_%s.txt' % (config.summary_path , time.strftime("%Y-%m-%d-%H-%M", time.localtime(startTime)))
            text = 'Train Step: %04d/%04d, Accuracy: %.2f, Text: %s\n' % (step, config.train_steps, accuracy, text)
            with open(exportFile, 'a') as a_writer:
                if step + 1 == config.sample_every:
                    a_writer.write('Original text: %s\n' % (startText))
                a_writer.write(text)

        if step == config.train_steps:
            break

    print('Done training.')


###############################################################################
###############################################################################

if __name__ == "__main__":

    # Parse training configuration
    parser = argparse.ArgumentParser()

    # Model params
    parser.add_argument('--txt_file', type=str, required=True,
                        help="Path to a .txt file to train on")
    parser.add_argument('--seq_length', type=int, default=30,
                        help='Length of an input sequence')
    parser.add_argument('--lstm_num_hidden', type=int, default=128,
                        help='Number of hidden units in the LSTM')
    parser.add_argument('--lstm_num_layers', type=int, default=2,
                        help='Number of LSTM layers in the model')

    # Training params
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=2e-3,
                        help='Learning rate')

    # It is not necessary to implement the following three params,
    # but it may help training.
    parser.add_argument('--learning_rate_decay', type=float, default=0.96,
                        help='Learning rate decay fraction')
    parser.add_argument('--learning_rate_step', type=int, default=5000,
                        help='Learning rate step')
    parser.add_argument('--dropout_keep_prob', type=float, default=1.0,
                        help='Dropout keep probability')

    parser.add_argument('--train_steps', type=int, default=int(1e6),
                        help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=5.0, help='--')

    # Misc params
    parser.add_argument('--summary_path', type=str, default="./summaries/",
                        help='Output path for summaries')
    parser.add_argument('--print_every', type=int, default=5,
                        help='How often to print training progress')
    parser.add_argument('--sample_every', type=int, default=100,
                        help='How often to sample from the model')
    parser.add_argument('--greed_sampling', type=int, default=0,
                        help='If greedy sampling should be applied')
    parser.add_argument('--device', type=str, default=("cpu" if not torch.cuda.is_available() else "cuda"),
                        help="Device to run the model on.")

    # If needed/wanted, feel free to add more arguments

    config = parser.parse_args()

    # Train the model
    train(config)

#python train.py --txt_file ./assets/book_EN_democracy_in_the_US.txt --train_steps 5000 --device cpu 
#python train.py --txt_file ./assets/book_EN_democracy_in_the_US.txt --train_steps 5000 --device cuda 