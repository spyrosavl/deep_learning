"""
This module implements training and evaluation of a multi-layer perceptron in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_pytorch import MLP
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100
NEG_SLOPE_DEFAULT = 0.02

# Directory in which cifar data is saved
DATA_DIR_DEFAULT = './cifar10/cifar-10-batches-py'

FLAGS = None


def accuracy(predictions, targets):
    """
    Computes the prediction accuracy, i.e. the average of correct predictions
    of the network.
    
    Args:
      predictions: 2D float array of size [batch_size, n_classes]
      labels: 2D int array of size [batch_size, n_classes]
              with one-hot encoding. Ground truth labels for
              each sample in the batch
    Returns:
      accuracy: scalar float, the accuracy of predictions,
                i.e. the average correct predictions over the whole batch
    
    TODO:
    Implement accuracy computation.
    """
    predictions = predictions.argmax(1)
    total = predictions.shape[0]
    correct = (predictions == targets).sum()
    return torch.true_divide(correct, total)
    

def train():
    """
    Performs training and evaluation of MLP model.
  
    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    #neg_slope = FLAGS.neg_slope
    mlp = MLP(32*32*3, dnn_hidden_units, 10)
    lossModule = nn.CrossEntropyLoss()
    optimizer = optim.SGD(mlp.parameters(), lr=FLAGS.learning_rate)
    #train
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    losses = []
    train_acc = []
    test_acc = []
    for step in range(FLAGS.max_steps):                                #for epoch
        data, targets = cifar10['train'].next_batch(FLAGS.batch_size)
        data, targets = torch.from_numpy(data.reshape(FLAGS.batch_size, -1)), torch.from_numpy(targets).argmax(1)
        optimizer.zero_grad()
        predictions = mlp.forward(data)                                 #forward pass
        loss = lossModule(predictions, targets)                         #calculate loss
        loss.backward()                                                 #backpropagation
        optimizer.step()                                                #update params
        #evaluation
        if step > 0 and step % FLAGS.eval_freq == 0:
            losses.append(loss.item())
            train_acc.append(accuracy(predictions, targets))
            dataTest, targetsTest = cifar10['test'].next_batch(5000)
            dataTest, targetsTest = torch.from_numpy(dataTest.reshape(5000, -1)), torch.from_numpy(targetsTest).argmax(1)
            predictionsTest = mlp.forward(dataTest)
            test_acc.append(accuracy(predictionsTest, targetsTest))
            print("Step: %d, Loss: %f, Train Accuracy: %f, Test Accuracy: %f" % (step, losses[-1], train_acc[-1], test_acc[-1]))
        
    plt.plot(np.arange(FLAGS.max_steps/FLAGS.eval_freq-1), losses, label='Cross Entropy Loss')
    plt.plot(np.arange(FLAGS.max_steps/FLAGS.eval_freq-1), train_acc, label='Accuracy (train)')
    plt.plot(np.arange(FLAGS.max_steps/FLAGS.eval_freq-1), test_acc, label='Accuracy (test)')
    plt.xlabel('training step')
    plt.legend()
    plt.show()


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(FLAGS).items():
        print(key + ' : ' + str(value))


def main():
    """
    Main function
    """
    # Print all Flags to confirm parameter settings
    print_flags()
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
    
    # Run the training operation
    train()


if __name__ == '__main__':
    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type=str, default=DNN_HIDDEN_UNITS_DEFAULT,
                        help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type=float, default=LEARNING_RATE_DEFAULT,
                        help='Learning rate')
    parser.add_argument('--max_steps', type=int, default=MAX_STEPS_DEFAULT,
                        help='Number of steps to run trainer.')
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT,
                        help='Batch size to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
    parser.add_argument('--data_dir', type=str, default=DATA_DIR_DEFAULT,
                        help='Directory for storing input data')
    FLAGS, unparsed = parser.parse_known_args()
    
    main()
