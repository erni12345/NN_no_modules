import numpy as np
from XORNeuralNet import Neural_Network

# Define the XOR input data and corresponding target values
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
Y = np.array([[0, 1, 1, 0]])

# Create a Neural Network with appropriate input, hidden, and output layer sizes
input_size = 2
hidden_size = 4  # You can adjust the size of the hidden layer if needed
output_size = 1
learning_rate = 0.1

# Create the Neural Network
nn = Neural_Network(input_size, hidden_size, output_size, learning_rate)

# Train the network on the XOR data
number_of_steps = 10000  # You can adjust the number of training steps
nn.learn(number_of_steps, X, Y)

# Test the trained network on XOR inputs
test_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
predictions = nn.predict(test_inputs)

# Print the predictions
print("Predictions for XOR input:")
for i in range(4):
    print(f"Input: {test_inputs[:, i]}, Prediction: {predictions[0, i]}")
