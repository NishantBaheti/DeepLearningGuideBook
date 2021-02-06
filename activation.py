"""Understanding activation functions 

Notes:

    1. Introducing non linearity to the network. Why?
    2. According to me we need one parameter to compare all the nodes results after learning
         and passing the value to upcoming nodes.
    3. To make sense of the data and a mapping for approximation.
    4. Understand what is the impact of weights and biases changing value 
        to the network/nodes.
        
        If there is only linear fx then it can only fit linear data but if we have not linear data 
        like a sine wave then it will fail to do so.


    5. If there is no activate function then the whole network will be similar to a one linear
        node.
        
        w.T(w.T *(w.T * x + b) + b) + b ... = output

    6. stepwise activation fx 

        * non granular 
        * only 0 and 1
        * more info on 

    7. sigmoid activation fx

        * granular
        * between 0 and 1
        * more info on  
        * Comparatively complex calcultaion

    8. ReLU activation fx 

        * granular
        * between 0 to x
        * more info on 
        * easy calculation 
        * almost linear but rectified so less than zeros are not allowed.
            so introducing slight non linearity makes it eligible for an
            activation function but also inherently easy and fast calculation 
            than sigmoid.

"""


from os import XATTR_REPLACE
import numpy as np

np.random.seed(0)  # everytime random result will be same based on seed

X = np.array([
    [1.0, 2.0, 3, 2.5],
    [2.0, 5.0, 1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]
])  # (3,4) (number of examples, number of features)


def ReLU(x):
    """
    """
    output = []

    for i in x:
        if i<0:
            output.append(0)
        else:
            output.append(x)

    return output 

def sigmoid(x):
    """
    """
    output = []

    for i in x:
        output.append(1/(1 + np.exp(-i)))

    return output

def stepwise(x):
    """
    """
    output = []
    for i in x:
        if i <= 0:
            output.append(0)
        else:
            output.append(1)
