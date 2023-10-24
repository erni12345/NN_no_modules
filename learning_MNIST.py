import tensorflow as tf
from NeuralNet import Neural_Network
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train_vectors = x_train.reshape(x_train.shape[0], -1).T  # Reshape to (60000, 784)
x_test_vectors = x_test.reshape(x_test.shape[0], -1).T


input_size = 784
hidden_size = 10
output_size = 10
learning_rate = 0.1

# Create the Neural Network
nn = Neural_Network(input_size, hidden_size, output_size, learning_rate)

# Train the network on the XOR data
number_of_steps = 500
nn.learn(number_of_steps, x_train_vectors, y_train)
"""
print(x_train)
image_index = 100
plt.imshow(x_train[image_index], cmap='gray')
plt.title(f'MNIST Digit: {y_train[image_index]}')
plt.show()"""
