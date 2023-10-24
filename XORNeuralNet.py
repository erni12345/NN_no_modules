import numpy as np

#for now we start with only one layer


#we are doing a 2 input -> 2 neurons -> one neuron to approximate the XOR function

class Neural_Network:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        """
        We initialize our Weights and Biases based on the layer sizes.
        """
        self.W1 = np.random.randn(hidden_size, input_size) - 0.5
        self.B1 = np.random.randn(hidden_size, 1) - 0.5
        self.W2 = np.random.randn(output_size, hidden_size) - 0.5
        self.B2 = np.random.randn(output_size, 1) - 0.5
        self.learning_rate = learning_rate

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def sigmoid_derivative(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def forward_propagation(self, X):
        # Input to hidden layer
        self.Z1 = np.dot(self.W1, X) + self.B1
        self.A1 = self.sigmoid(self.Z1)
        
        # Hidden layer to output
        self.Z2 = np.dot(self.W2, self.A1) + self.B2
        self.A2 = self.sigmoid(self.Z2)

    def backpropagation(self, X, Y):
        # Calculate the loss
        loss = 0.5 * np.square(self.A2 - Y)
        
        # Compute delta for the output layer
        delta2 = (self.A2 - Y) * self.sigmoid_derivative(self.A2)
        dW2 = np.dot(delta2, self.A1.T)
        db2 = delta2
        
        # Compute delta for the hidden layer
        delta1 = np.dot(self.W2.T, delta2) * self.sigmoid_derivative(self.A1)
        dW1 = np.dot(delta1, X.T)
        db1 = delta1
        
        return loss, dW1, db1, dW2, db2

    def update_weights(self, dW1, db1, dW2, db2):
        self.W1 = self.W1 - self.learning_rate * dW1
        self.B1 = self.B1 - self.learning_rate * db1
        self.W2 = self.W2 - self.learning_rate * dW2
        self.B2 = self.B2 - self.learning_rate * db2

    def learn(self, number_of_steps, X, Y):
        for x in range(number_of_steps):
            self.forward_propagation(X)
            loss, dW1, db1, dW2, db2 = self.backpropagation(X, Y)
            self.update_weights(dW1, db1, dW2, db2)

            if x % 10 == 0:
                total_loss = np.mean(loss)
                print(f'step : {x}')
                print("Total Average loss = ", total_loss)

    def predict(self, X):
        self.forward_propagation(X)
        return self.A2

    def save(self):
        pass
