"""Multiple Neuron with Numpy multiplication

Notes:

    1.         
                weight---> 
        input -----------> [||||] ----> output
                bias--->

        output = weight * input + bias
"""


import numpy as np 


# 3 neurons 4 inputs 1 output
inputs = [1.0, 2.0, 3, 2.5] #(4,)

weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
] #(3,4)


biases = [2, 3, 0.5] #(3,)

output = np.dot(weights,inputs) + biases # (3,4)dot(4,) + (3,) = (3,)
print(output)


###############################################
########## with vectors to see the basic idea 
# just because both weights and inputs are vector
# it wont matter what the sequence is . it will be vector multiplication
# it matters only when these are matrices or tensors
# which will be real world scenario

# inputs = [1.0, 2.0, 3, 2.5]
# weights = [0.2, 0.8, -0.5, 1.0]
# bias = 2

# output = np.dot(weights,inputs) + bias

# print(output)




 
