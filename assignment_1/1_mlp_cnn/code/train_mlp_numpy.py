"""
This module implements training and evaluation of a multi-layer perceptron in NumPy.
You should fill in code into indicated sections.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropyModule
import cifar10_utils

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
#MAX_STEPS_DEFAULT = 1400
MAX_STEPS_DEFAULT= 5
BATCH_SIZE_DEFAULT = 200
#EVAL_FREQ_DEFAULT = 100
EVAL_FREQ_DEFAULT = 1

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
    predictions = np.vstack(predictions)
    targets = np.vstack(targets)
    predictions = np.argmax(predictions, 1)
    targets = np.argmax(targets, 1)
    total = predictions.shape[0]
    correct = (predictions == targets).sum()
    return correct/total

def step(mlp):
    for layer in mlp.layers:
        layer.params['weight'] -= FLAGS.learning_rate * layer.grads['weight']
        layer.params['bias'] -= FLAGS.learning_rate * layer.grads['bias']

def train():
    """
    Performs training and evaluation of MLP model.

    TODO:
    Implement training and evaluation of MLP model. Evaluate your model on the whole test set each eval_freq iterations.
    """

    ### DO NOT CHANGE SEEDS!
    # Set the random seeds for reproducibility
    np.random.seed(42)

    ## Prepare all functions
    # Get number of units in each hidden layer specified in the string such as 100,100
    if FLAGS.dnn_hidden_units:
        dnn_hidden_units = FLAGS.dnn_hidden_units.split(",")
        dnn_hidden_units = [int(dnn_hidden_unit_) for dnn_hidden_unit_ in dnn_hidden_units]
    else:
        dnn_hidden_units = []
    
    mlp = MLP(32*32, dnn_hidden_units, 10)
    lossModule = CrossEntropyModule()
    #train
    cifar10 = cifar10_utils.get_cifar10(DATA_DIR_DEFAULT)
    data, targets = cifar10['train'].next_batch(FLAGS.batch_size)
    for step in range (FLAGS.max_steps):                                #for epoch
        predictions = mlp.forward(data)                                 #forward pass
        loss = lossModule.forward(predictions, targets)                 #calculate loss
        mlp.backward(loss)                                              #backpropagation
        step(mlp)                                                       #update params
        data, targets = cifar10['train'].next_batch(FLAGS.batch_size)   # get data for next batch

        #evaluation
        # if step > 0 and step % FLAGS.eval_freq == 0:
        #     dataTest, targetsTest = cifar10['train'].next_batch(5000)
        #     predictionsTest = mlp.forward(dataTest)
        #     acc = accuracy(predictionsTest, targetsTest)
        #     print("Step: %d, Accuracy: %f" % (step, acc))



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
