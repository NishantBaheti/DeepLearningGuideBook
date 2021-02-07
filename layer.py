"""Undestanding feed forward.

* Shapes 
    
    * According to Andrew Ng's course :

        * inputs  : ( m examples, number of input features)
        * weights : ( number of nodes in previous layer or input features, number of nodes in current layer )
        * biases  : ( number of nodes in current layer , 1 )

        - output = weights.T * inputs + biases 
        - (n[l],m) = (n[l],n[l-1]) * (n[l-1],m) + (n[l],1)

    * According to Sentdex's example :

        * inputs  : ( m examples, number of input features )
        * weights : ( number of nodes in previous layer or input features, number of nodes in current layer )
        * biases  : ( 1, number of nodes in current layer )

        - output = weights.T * inputs + biases
        - (n[l],m) = (n[l],n[l-1]) * (n[l-1],m) + (1 , n[l])

==========================================================================================================
"""


import numpy as np

np.random.seed(0) # everytime random result will be same based on seed

X = np.array([
    [1.0, 2.0, 3, 2.5],
    [2.0, 5.0,1.0, 2.0],
    [-1.5,2.7,3.3,-0.8]
    ])  # (3,4)

print(X.shape)

class Layer_Dense:
    """Layer Module
    
    It is recommended that input data X is scaled(data scaling operations)
    so that data is normalized but meaning of the data remains same.
    
    Notes:
       How do we actually initialize a layer for a New Neural Network?
        
        1. initialize weights with small random values
            why? because according to Andrew Ng's explanation if all the weights/params are
            initialized by zero or same value then all the hidden units will be symmetric with identical nodes.
            
            With identical nodes there will be no learning/ decision making. because all the decisions
            shares same value.
            
            If all the nodes will have zero values(weights are zero , multiplication with weights will also be 
            zero) and propogation result wont be a conclusive one(dead network).

            shape of weights(theoratically) : (number of neurons, number of inputs)
                                                but we have to do transpose operation everytime
            shape of weights(for code) : (number of inputs, number of neurons)

            number of inputs = number of neurons in previous layer or input layer features
            
        2. initialize of bias can be zero. as randomness is already introduced by weights.
            But for smaller Neural Network it is advised to not to initialize with zero.

            
            shape of biases (sentdex) : (1, number of neurons)
            shape of biases (Andrew Ng) : (number of neurons,1) 

            don't really know which one is more correct

            In the both are going to broadcasted to the base result 
            

    Attributes:
        n_inputs (int) : number of inputs 
        n_neurons (int) : number of neurons

    """
    def __init__(self,n_inputs,n_neurons):
        """
        """
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons) # multiply by 0.1 to make it small
        self.biases = np.zeros((1,n_neurons))
        self.output = None

    def forward(self, inputs):
        """forward propogation calculation
        
        Method:
            output = inputs * weights + biases
        
        Args:
            inputs (numpy.ndarray) : X Input matrix
        
        """
        self.output = np.dot(inputs,self.weights)+self.biases 
    

class Activation_ReLU:
    """ReLU Activation Fx

    Method:
            
        f(x) = 0  if x <= 0
        f(x) = x  if x > 0
    
    """
    def __init__(self):
        """
        """
        self.output = None

    def forward(self, inputs):
        """Apply ReLU to input
        
        Args:
            inputs (numpy.ndarray) : input matrix
        
        """
        self.output = np.maximum(0,inputs)


class Activation_Softmax:
    """
    """
    def forward(self,inputs):
        """
        """
        exp_values = np.exp(inputs - np.max(inputs,axis=1,keepdims=True))
        probabilites = exp_values  / np.sum(exp_values,axis=1,keepdims=True)
        self.output = probabilites


layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()

layer1.forward(X)
activation1.forward(layer1.output)


layer2.forward(activation1.output)
activation2.forward(layer2.output)

print(activation2.output)



