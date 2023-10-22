import numpy as np

#for now we start with only one layer


#we are doing a 2 input -> 2 neurons -> one neuron to approximate the XOR function

class Neural_Network:

    

    def __init__(self, size_layer_1, size_layer_2, X_train, Y_train, learning_rate) -> None:
        """
            We initialize our Weights for layer 1, bias for layer 1, weigths for layer 2 and bias for layer 2
        """
        self.W1 = np.random.randn(2, 2) - 0.5
        self.B1 = np.random.randn(2, 1) - 0.5
        self.W2 = np.random.randn(1, 2) - 0.5
        self.B2 = np.random.randn(1, 1) - 0.5
        self.X_train = X_train
        self.Y_train = Y_train
        self.learning_rate = learning_rate


    def sigmoid(self, z):
            return 1 / (1 + np.exp(-z))
    

    def forward_propagation(self):
        
        self.Z1 = self.W1.dot(self.X_train)
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.W2.dot(self.A1)
        self.A2 = self.sigmoid(self.Z2)
        return self.Z1, self.A1, self.Z2, self.A2

    def back_propagation(self):

        dL = self.A2 - self.Y_train
        dA2 = self.A2 * (1 - self.A2)
        dZ2 = self.W2
        dA1 = self.A1 * (1 - self.A1)
        dZ1 = self.W1

        print(self.W1)
        print("____")
        print(self.Z1)
        print("____")
        print(self.Z2)
        print("____")
        dLW2 = dZ2 * dA2 * dL
        dLW1 = dZ1 * dA1 * dLW2
        
        self.W1 = self.W1 - self.learning_rate * dLW1
        self.W2 = self.W1 - self.learning_rate * dLW2


         




NN = Neural_Network(2, 1, np.array([[1, 1],[1, 0],[0, 1],[0, 0]]).T, np.array([0, 1, 1, 0]), 0.01)

NN.forward_propagation()
NN.back_propagation()