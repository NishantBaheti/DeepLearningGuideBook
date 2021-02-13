"""Understanding activation functions 

Notes:

    1. Introducing non linearity to the network. Why?
    2. According to me we need one parameter to compare all the nodes results after learning and passing the value to upcoming nodes.
    3. To make sense of the data and a mapping for approximation.
    4. Understand what is the impact of weights and biases changing value to the network/nodes.
        If there is only linear fx then it can only fit linear data but if we have not linear data like a sine wave then it will fail to do so. 
    5. If there is no activate function then the whole network will be similar to a one linear node.
        
            w.T(w.T *(w.T * x + b) + b) + b ... = output 
            
=====================================================================================================
"""
import numpy as np

class Activation_Stepwise:
    """Stepwise Activation Fx

    Notes:

        * non granular 
        * only 0 and 1
        
    References:
        None
    """

    def __init__(self):
        """
        """
        self.output = None

    def forward(self, inputs):
        """Apply Stepwise to inputs

        Args:
            inputs (numpy.ndarray) : input matrix

        """
        pass


class Activation_Sigmoid:
    """Sigmoid Activation Fx

    Notes:

        f(x) = 1 / (1 + e^-x)

        * granular
        * between 0 and 1
        * Comparatively complex calcultaion

    References:
        None
    """

    def __init__(self):
        """
        """
        self.output = None

    def forward(self, inputs):
        """Apply Sigmoid to input
        
        Args:
            inputs (numpy.ndarray) : input matrix
        
        """
        self.output = 1 / (1 + np.exp(-inputs))

    def backward(self, dvalues):
        """
        """
        pass


class Activation_ReLU:
    """ReLU Activation Fx

    Notes:
            
        f(x) = 0  if x <= 0
        f(x) = x  if x > 0

        * granular
        * between 0 to x
        * easy calculation 
        * almost linear but rectified so less than zeros are not allowed.so introducing slight non linearity makes it eligible for an activation function but also inherently easy and fast calculation than sigmoid.
    
    References:
        None
    """

    def __init__(self):
        """
        """
        self.inputs = None
        self.output = None
        self.dinputs = None 

    def forward(self, inputs):
        """Apply ReLU to input
        
        Args:
            inputs (numpy.ndarray) : input matrix
        
        """
        self.inputs = inputs # save inputs 
        self.output = np.maximum(0, inputs) # calculate from inputs

    def backward(self, dvalues):
        """Apply backward propogation

        Args:
            dvalues (numpy.ndarray) : inputs from previous later in backward prop
        
        Notes:
            None
        """
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0] = 0


        


class Activation_Softmax:
    """
    """

    def forward(self, inputs):
        """Forward propogation calculation

        Args:
            inputs (numpy.ndarray) : input matrix

        Notes:
            TODO
        """
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))
        probabilites = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilites

    def backward(self, dvalues):
        """backward pass

        Args:
            dvalues (numpy.ndarray) : gradient values

        Notes:
            None
        """
        pass
        
