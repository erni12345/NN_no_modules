import numpy as np
import matplotlib.pyplot as plt
#for now we start with only one layer

#for now only one hidden layer

class Neural_Network:
    def __init__(self, input_size, hidden_size, output_size, learning_rate):
        """
        We initialize our Weights and Biases based on the layer sizes.
        """
        self.output_size = output_size
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


    def one_hot_encode(self, labels, num_classes):
        
        one_hot = np.zeros((len(labels), num_classes))
        
        for i in range(len(labels)):
            one_hot[i, labels[i]] = 1
        
        return one_hot
    

    def backpropagation(self, X, Y):
        # Calculate the loss
        one_Y = self.one_hot_encode(Y, self.output_size)

        loss = 0.5 * np.square(self.A2.T - one_Y)
        
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
        loss_overall = []
        for x in range(number_of_steps):
            self.forward_propagation(X)
            loss, dW1, db1, dW2, db2 = self.backpropagation(X, Y)
            self.update_weights(dW1, db1, dW2, db2)

            if x % 10 == 0:

                total_loss = np.mean(loss)
                print(f'step : {x}')
                print("Total Average loss = ", total_loss)
                loss_overall.append(total_loss)

        self.visualize_learning(loss_overall)

    def predict(self, X):
        self.forward_propagation(X)
        return self.A2


    def visualize_learning(self, loss_overall):

        plt.plot(loss_overall)
        plt.show()




    def save(self, filename):

        data = {
            "W1": self.W1,
            "B1": self.B1,
            "W2": self.W2,
            "B2": self.B2
        }
        np.savez(filename, **data)

    @classmethod
    def load(cls, filename):

        loaded_data = np.load(filename)
        W1 = loaded_data["W1"]
        B1 = loaded_data["B1"]
        W2 = loaded_data["W2"]
        B2 = loaded_data["B2"]

        # Determine the layer sizes from the loaded weights
        input_size, hidden_size = W1.shape[1], W1.shape[0]
        output_size = W2.shape[0]

        # Create a new instance of Neural_Network with the loaded weights
        nn = cls(input_size, hidden_size, output_size, learning_rate=0)  # Set learning_rate to 0
        nn.W1 = W1
        nn.B1 = B1
        nn.W2 = W2
        nn.B2 = B2

        return nn

