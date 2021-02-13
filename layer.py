"""Undestanding feed forward.

* Shapes 
    
    * According to Andrew Ng's course :

        * inputs  : 
            ( m examples, number of input features )
        * weights : 
            ( number of nodes in previous layer or input features, number of nodes in current layer )
        * biases  : 
            ( number of nodes in current layer , 1 )

        - output = weights.T * inputs + biases 
        - ( n[l] ,m ) = ( n[l] , n[l-1] ) * ( n[l-1] , m ) + ( n[l] , 1 )

    * According to Sentdex's example :

        * inputs  : 
            ( m examples, number of input features )
        * weights : 
            ( number of nodes in previous layer or input features, number of nodes in current layer )
        * biases  : 
            ( 1, number of nodes in current layer )

        - output = weights.T * inputs + biases
        - ( n[l] , m ) = ( n[l] , n[l-1] ) * ( n[l-1] , m ) + ( 1 , n[l] )

==========================================================================================================
"""

from abc import ABC,abstractmethod
import numpy as np
from activation import Activation_Softmax,Activation_ReLU

np.random.seed(0) # everytime random result will be same based on seed

X = np.array([
    [1.0, 2.0, 3, 2.5],
    [2.0, 5.0,1.0, 2.0],
    [-1.5,2.7,3.3,-0.8],
    [1.5,2.0,1.2,3.3]
    ])  # (4,4)

y = np.array([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,0],
    [0,0,0,1]
])

class Layer_Dense:
    """Layer Module
    
    It is recommended that input data X is scaled(data scaling operations)
    so that data is normalized but meaning of the data remains same.

    Attributes:
        n_inputs (int) : number of inputs 
        n_neurons (int) : number of neurons
    
    Notes:
       How do we actually initialize a layer for a New Neural Network?
        
        1. initialize weights with small random values
            
            why? because according to Andrew Ng's explanation if all the weights/params are
            initialized by zero or same value then all the hidden units will be symmetric with identical nodes.
            
            With identical nodes there will be no learning/ decision making. because all the decisions
            shares same value.
            
            If all the nodes will have zero values(weights are zero , multiplication with weights will also be 
            zero) and propogation result wont be a conclusive one(dead network).

            shape of weights(theoratically) : 
                    (number of neurons, number of inputs)  
                    but we have to do transpose operation everytime

            shape of weights(for code) : 
                    (number of inputs, number of neurons)

            number of inputs : 
                    number of neurons in previous layer or input layer features
            
        2. initialize of bias can be zero. 
            
            as randomness is already introduced by weights.
            But for smaller Neural Network it is advised to not to initialize with zero.

            
            shape of biases (sentdex) : 
                (1, number of neurons)

            shape of biases (Andrew Ng) : 
                (number of neurons,1) 

            don't really know which one is more correct
            In the end both are going to be broadcasted to the base result 
            
    """
    def __init__(self,n_inputs,n_neurons):
        """
        """
        self.weights = 0.10 * np.random.randn(n_inputs,n_neurons) # multiply by 0.1 to make it small
        self.biases = np.zeros((1,n_neurons))
        self.inputs = None 
        self.output = None
        self.dweights = None 
        self.dbiases = None
        self.dinputs = None 

    def forward(self, inputs):
        """forward propogation calculation
        
        Args:
            inputs (numpy.ndarray) : X Input matrix

        Notes:
            output = inputs * weights + biases
        """
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights)+self.biases 

    def backward(self, dvalues):
        """backward pass

        Args:
            dvalues (numpy.ndarray) : gradient value from the next layer to update this layers parameters

        Notes:
            
            Based on Andrew Ng's-

                input to this layer 
                    (for backward propogation)
                dZ' `dvalues` = A - y 
                    (basically difference or inaccuracy or loss on target value)
                
                param for this layer 
                    (this function starts working from here)
                dW = dZ * A.T
                dB = sum(dZ)

                input for next layer 
                    (in backward propogation)
                dZ = dZ' * W.T 

        """
        # gradient on parameters 
        self.dweights = np.dot(self.inputs.T * dvalues)
        self.dbiases = np.sum(dvalues, axis=0, keepdims=True)
        
        # gradient on values / input to next layer in backpropogation
        self.dinputs = np.dot(dvalues, self.weights.T)


class Loss(ABC):
    """Loss Meta class 

    """
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def forward(self):
        """mandatory method for child class
        """
        pass

    def calculate(self, output, y):
        """Calculate mean loss
        
        Args:
            output () : output from the layer
            y () : truth value/ target/ expected outcome

        Returns:
            None

        Notes:
            TODO
        
        References:
            None
        """
        sample_losses = self.forward(output, y) # it can be individual outcome of different kind of loss functions

        data_loss = np.mean(sample_losses) # calculating mean

        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    """Categorical Cross entropy loss 

    Notes:
        TODO
    """
    def __init__(self):
        pass

    def forward(self, y_pred, y_true):
        """forward propogation calculation 

        Args:
            y_pred (numpy.ndarray) : predictions generated
            y_true (numpy.ndarray) : actual values
        
        Notes:
            
            y_pred_clipped
                numpy.clip is used to clip the values from min
                and max values like bandpass filter
                min = 1.0 * 10^-7 
                max = 1 - 1.0 * 10^-7

            correct_confidences 
                probabilities for target value that has been 
                calculated earlier 
                only for categorical variables
                TODO : to write more about this 

            negative_log_Likelihoods
                TODO

        """

        # get total number of rows/samples
        samples = len(y_pred)

        
        y_pred_clipped = np.clip(y_pred,1e-7,1-1e-7)
        
        correct_confidences = None
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[
                range(samples),
                y_true
            ]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(
                y_pred_clipped * y_true,
                axis = 1
            )
        
        else:
            pass
        
        # losses
        negative_log_Likelihoods = -np.log(correct_confidences)
        return negative_log_Likelihoods



if __name__ == "__main__":
    dense1 = Layer_Dense(4,5) # hidden layer 1 with 4 inputs and 5 neurons 
    activation1 = Activation_ReLU() # layer1 activation function



    dense2 = Layer_Dense(5,4) # hidden layer 2 with 5 inputs and 4 neurons
    activation2 = Activation_Softmax() # layer 2 activation function


    loss_function = Loss_CategoricalCrossEntropy() # loss function for this network training


    # Helper variables for model
    lowest_loss = 1e+7
    best_dense1_weigths = dense1.weights.copy()
    best_dense1_biases = dense1.biases.copy()
    best_dense2_weights = dense2.weights.copy()
    best_dense2_biases = dense2.biases.copy()

    iterations = 10000

    for iteration in range(iterations):

        ############### here new code will come #######################
        dense1.weights += 0.05 * np.random.randn(4, 5)
        dense1.biases += 0.05 * np.random.randn(1, 5)
        dense2.weights += 0.05 * np.random.randn(5, 4)
        dense2.biases += 0.05 * np.random.randn(1, 4)
        ###############################################################


        dense1.forward(X)  # forward prop for layer 1
        activation1.forward(dense1.output)  # apply activation 1


        dense2.forward(activation1.output)  # forward prop for layer 2
        activation2.forward(dense2.output)  # apply activation 2

        # calculate loss 
        loss = loss_function.calculate(activation2.output, y)

        # calculate predictions 
        # np.argmax returns maximum value over an axis
        # so it will return max value from the axis
        predictions = np.argmax(activation2.output, axis=1)
    

        # calcuate accuracy
        # mean of all the matching predictions to the original 
        # target value
        # basic accuracy calculation
        accuracy = np.mean(predictions == y)

        # if loss is less than move these hyper parameters to best one yet
        if loss < lowest_loss:
            print('New set of weights found, iteration:', iteration,
                'loss:', loss, 'acc:', accuracy)
            best_dense1_weights = dense1.weights.copy()
            best_dense1_biases = dense1.biases.copy()
            best_dense2_weights = dense2.weights.copy()
            best_dense2_biases = dense2.biases.copy()
            lowest_loss = loss
            print("--------------------------------------------------")

        # Revert weights and biases if loss is not less than previous one 
        # go back last good one 
        else:
            dense1.weights = best_dense1_weights.copy()
            dense1.biases = best_dense1_biases.copy()
            dense2.weights = best_dense2_weights.copy()
            dense2.biases = best_dense2_biases.copy()

    
    



