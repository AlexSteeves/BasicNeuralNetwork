import numpy as np


arr_1d = np.array([1, 2, 3, 4, 5])
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])


input_data = np.array([[0,0], [0,1], [1,0], [1,1]])

output_data = np.array([[0], [1], [1], [0]])


#weights from input to hidden layer
weights_input_hidden = np.random.rand(2,4)
#Weights from hidden to output layer
weights_hidden_output = np.random.rand(4,1)


#Bias for hidden layer
bias_hidden = np.random.rand(4)
#Bias for output layer
bias_output = np.random.rand(1)


# function which defines the sigmoid activation
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Function that defines the sigmoid derivative
def sigmoid_derivative(x):
    return x * (1 - x)

epochs = 10000
learning_rate = 0.1

for epoch in range(epochs):

    # Forward propagation
    hidden_layer_input = np.dot(input_data, weights_input_hidden) + bias_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)

    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + bias_output
    predicted_output = sigmoid(output_layer_input)

    # Calculate the error
    error = output_data - predicted_output

    # Backpropagation
    d_predicted_output = error * sigmoid_derivative(predicted_output)
    error_hidden_layer = d_predicted_output.dot(weights_hidden_output.T)
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_predicted_output) * learning_rate
    weights_input_hidden += input_data.T.dot(d_hidden_layer) * learning_rate

    bias_output += np.sum(d_predicted_output) * learning_rate
    bias_hidden += np.sum(d_hidden_layer) * learning_rate

# Test the trained neural network
test_input = np.array([[1, 0]])
hidden_layer_test = sigmoid(np.dot(test_input, weights_input_hidden) + bias_hidden)
output = sigmoid(np.dot(hidden_layer_test, weights_hidden_output) + bias_output)

print("Predicted output:", output)