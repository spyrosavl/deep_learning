"""
This module implements training and evaluation of a Convolutional Neural Network in PyTorch.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from convnet_pytorch import ConvNet
import cifar10_utils
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_STEPS_DEFAULT = 5000
EVAL_FREQ_DEFAULT = 500
OPTIMIZER_DEFAULT = 'ADAM'

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
    Performs training and evaluation of ConvNet model.
  
    TODO:
    Implement training and evaluation of ConvNet model. Evaluate your model on the whole test set each eval_freq iterations.
    """
    
    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    n_classes = cifar10['train'].labels.shape[1]
    convNet = ConvNet(3, n_classes).to(device)
    lossModule = nn.CrossEntropyLoss()
    optimizer = optim.Adam(convNet.parameters(), lr=FLAGS.learning_rate)
    #train
    losses = []
    train_acc = []
    test_acc = []
    for step in range(FLAGS.max_steps):                                #for epoch
        data, targets = cifar10['train'].next_batch(FLAGS.batch_size)
        data, targets = torch.from_numpy(data).to(device), torch.from_numpy(targets).argmax(1).to(device)
        optimizer.zero_grad()
        predictions = convNet.forward(data)                                 #forward pass
        loss = lossModule(predictions, targets)                         #calculate loss
        loss.backward()                                                 #backpropagation
        optimizer.step()                                                #update params
        #evaluation
        if step > 0 and step % FLAGS.eval_freq == 0:
            losses.append(loss.item())
            train_acc.append(accuracy(predictions, targets))
            dataTest, targetsTest = cifar10['test'].images, cifar10['test'].labels
            dataTest, targetsTest = torch.from_numpy(dataTest).to(device), torch.from_numpy(targetsTest).argmax(1).to(device)
            predictionsTest = convNet.forward(dataTest)
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
