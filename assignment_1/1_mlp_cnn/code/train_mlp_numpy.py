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
import matplotlib.pyplot as plt

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '100'
LEARNING_RATE_DEFAULT = 1e-3
MAX_STEPS_DEFAULT = 1400
BATCH_SIZE_DEFAULT = 200
EVAL_FREQ_DEFAULT = 100

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
    predictions = np.argmax(predictions, 1)
    targets = np.argmax(targets, 1)
    total = predictions.shape[0]
    correct = (predictions == targets).sum()
    return correct/total

def updateParams(mlp):
    for layer in mlp.layers:
        if hasattr(layer, 'params'):
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
    
    cifar10 = cifar10_utils.get_cifar10(FLAGS.data_dir)
    x, y, z = cifar10['train'].images.shape[1:]
    n_classes = cifar10['train'].labels.shape[1]
    mlp = MLP(x*y*z, dnn_hidden_units, n_classes)
    lossModule = CrossEntropyModule()
    #train
    loss = []
    train_acc = []
    test_acc = []
    for step in range (FLAGS.max_steps):                                #for epoch
        data, targets = cifar10['train'].next_batch(FLAGS.batch_size)
        data = data.reshape(FLAGS.batch_size, -1)
        predictions = mlp.forward(data)                                 #forward pass
        lossTMP = lossModule.forward(predictions, targets)              #calculate loss
        lossGrad = lossModule.backward(predictions, targets)
        mlp.backward(lossGrad)                                          #backpropagation
        updateParams(mlp)                                               #update params

        #evaluation
        if step > 0 and step % FLAGS.eval_freq == 0:
            loss.append(lossTMP)
            train_acc.append(accuracy(predictions, targets))
            dataTest, targetsTest = cifar10['test'].next_batch(5000)
            dataTest = dataTest.reshape(5000, -1)
            predictionsTest = mlp.forward(dataTest)
            test_acc.append(accuracy(predictionsTest, targetsTest))
            print("Step: %d, Loss: %f, Train Accuracy: %f, Test Accuracy: %f" % (step, loss[-1], train_acc[-1], test_acc[-1]))
        

    plt.plot(np.arange(FLAGS.max_steps/FLAGS.eval_freq-1), loss, label='Cross Entropy Loss')
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
