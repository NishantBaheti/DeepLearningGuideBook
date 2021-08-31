"""Batch wise multiple neurons

Notes:
    1. Helps with generalizations.
        example.- batch of samples from sensor data can create more generalized fitting
        instead of fitting for one data point at a time it can fit to a whole batch.

        More basic example would be seeing one example at a time to learn something
        or multiple examples at a time to learn something. Multiple examples (batch) would give 
        more general idea and less movement for learning. 

    2. bigger the batch , more number of parallel operations we can run

"""

import numpy as np


# 3 neurons 4 inputs 1 output
inputs = np.array([
    [1.0, 2.0, 3, 2.5],
    [2.0, 5.0,1.0, 2.0],
    [-1.5,2.7,3.3,-0.8]
    ])  # (3,4)

# layer1

weights1 = np.array([
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
])  # (3,4)


biases1 = np.array([2, 3, 0.5]).reshape(1,-1)  # (3,1)

# layer2

weights2 = np.array([
    [0.1, -0.14, 0.5],
    [-0.5,0.12,-0.33], 
    [-0.44,0.73,-0.13]
])  # (3,3)


biases2 = np.array([-1,2,-0.5]).reshape(1, -1)  # (3,1)


layer1_output = np.dot(inputs, weights1.T) + \
    biases1  # (3,4)dot(4,3) + (3,1) = (3,3)

layer2_output = np.dot(layer1_output, weights2.T) + \
    biases2  # (3,3)dot(3,3) + (3,1) = (3,3)

print(layer2_output)



