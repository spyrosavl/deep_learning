"""
This module implements various modules of the network.
You should fill in code into indicated sections.
"""
import numpy as np

class LinearModule(object):
    """
    Linear module. Applies a linear transformation to the input data.
    """
    
    def __init__(self, in_features, out_features):
        """
        Initializes the parameters of the module.
    
        Args:
          in_features: size of each input sample
          out_features: size of each output sample
    
        TODO:
        Initialize weights self.params['weight'] using normal distribution with mean = 0 and
        std = 0.0001. Initialize biases self.params['bias'] with 0.
    
        Also, initialize gradients with zeros.
        """
        self.params = {'weight': np.random.normal(0, 0.0001, (in_features, out_features)), 'bias' : np.zeros(out_features)}
        self.grads = {'weight': np.zeros((in_features, out_features)), 'bias' : np.zeros(out_features)}
    
    def forward(self, x):
        """
        Forward pass.
    
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        self.x = x #to be used in backward
        out = x @ self.params['weight'].T + self.params['bias']
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
    
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to
        layer parameters in self.grads['weight'] and self.grads['bias'].
        """
        
        self.grads['weight'] = self.x.T @ dout
        self.grads['bias'] = dout
        dx = dout @ self.params['weight']
        return dx



class SoftMaxModule(object):
    """
    Softmax activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.
        Args:
          x: input to the module
        Returns:
          out: output of the module
    
        TODO:
        Implement forward pass of the module.
        To stabilize computation you should use the so-called Max Trick - https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    
        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        raise NotImplementedError
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous modul
        Returns:
          dx: gradients with respect to the input of the module
    
        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        raise NotImplementedError
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return dx


class CrossEntropyModule(object):
    """
    Cross entropy loss module.
    """
    
    def forward(self, x, y):
        """
        Forward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          out: cross entropy loss
    
        TODO:
        Implement forward pass of the module.
        """
      
        N = x.shape[0]
        out = -np.sum(y*np.log(x+1e-9))/N
        
        return out
    
    def backward(self, x, y):
        """
        Backward pass.
        Args:
          x: input to the module
          y: labels of the input
        Returns:
          dx: gradient of the loss with the respect to the input x.
    
        TODO:
        Implement backward pass of the module.
        """
        
        dx = - (y / x) / y.shape[0]
        
        return dx


class ELUModule(object):
    """
    ELU activation module.
    """
    
    def forward(self, x):
        """
        Forward pass.

        Args:
          x: input to the module
        Returns:
          out: output of the module

        TODO:
        Implement forward pass of the module.

        Hint: You can store intermediate variables inside the object. They can be used in backward pass computation.
        """
        ########################
        # PUT YOUR CODE HERE  #
        #######################
        
        raise NotImplementedError
        
        ########################
        # END OF YOUR CODE    #
        #######################
        
        return out
    
    def backward(self, dout):
        """
        Backward pass.
        Args:
          dout: gradients of the previous module
        Returns:
          dx: gradients with respect to the input of the module

        TODO:
        Implement backward pass of the module.
        """
        
        ########################
        # PUT YOUR CODE HERE  #
        #######################

        raise NotImplementedError

        ########################
        # END OF YOUR CODE    #
        #######################
        return dx
