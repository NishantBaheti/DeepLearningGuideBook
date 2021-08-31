# 3 neurons 4 inputs 1 output

inputs = [1.0, 2.0, 3, 2.5]

weights = [
    [0.2, 0.8, -0.5, 1.0], 
    [0.5, -0.91, 0.26, -0.5], 
    [-0.26, -0.27, 0.17, 0.87]
    ]
bias = [ 2, 3 , 0.5] 

layer_outputs = [] #output of the layer
# iteration starting from weigths and bias input by input 
for neuron_weights, neuron_bias in zip(weights,bias):
    neuron_output = 0 #output for a neuron  
    #iteration of input with associative weight
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input * weight

    neuron_output += neuron_bias # add bias for respective weight and input 
    layer_outputs.append(neuron_output)

print(layer_outputs)


####################################3
## Naive approach for understanding
# weights1 = [0.2, 0.8, -0.5, 1.0]
# weights2 = [0.5, -0.91, 0.26, -0.5]
# weights3 = [-0.26, -0.27, 0.17, 0.87]

# bias1 = 2
# bias2 = 3
# bias3 = 0.5

# output = [inputs[0]*weights1[0] + inputs[1]*weights1[1] + inputs[2]*weights1[2] + inputs[3]*weights1[3] + bias1,
#           inputs[0]*weights2[0] + inputs[1]*weights2[1] + inputs[2]*weights2[2] + inputs[3]*weights2[3] + bias2,
#           inputs[0]*weights3[0] + inputs[1]*weights3[1] + inputs[2]*weights3[2] + inputs[3]*weights3[3] + bias3]

# print(output)
