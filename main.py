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
    
    def sigmoid_derivative(self, z):
         
        return self.sigmoid(z) * (1 - self.sigmoid(z))


    def forward_propagation(self):
        
        self.Z1 = self.W1.dot(self.X_train) + self.B1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.W2.dot(self.A1)
        self.A2 = self.sigmoid(self.Z2) + self.B2
        return self.Z1, self.A1, self.Z2, self.A2

    def back_propagation(self):

        loss = 0.5 * np.square(self.A2 - self.Y_train)
        self.loss = loss

        delta2 = (self.A2 - self.Y_train) * self.sigmoid_derivative(self.Z2)
        dW2 =  np.dot(delta2, self.A1.T)
        db2 = delta2

        delta1 = np.dot(self.W2.T, delta2) * self.sigmoid_derivative(self.Z1)
        dW1 = np.dot(delta1, self.X_train.T)
        db1 = delta1

        self.dW1 = dW1
        self.dW2 = dW2
        self.db1 = db1
        self.db2 = db2


    def update_weights(self):
        
        self.W1 = self.W1 - self.learning_rate * self.dW1
        self.B1 = self.B1 - self.learning_rate * self.db1
        self.W2 = self.W1 - self.learning_rate * self.dW2
        self.B2 = self.B2 - self.learning_rate * self.db2

    def learn(self, number_of_steps):
         
        for x in range(number_of_steps):
            self.forward_propagation()
            self.back_propagation()
            self.update_weights()
            
            if(x % 10 == 0):
                print("Step : ", x)
                print(f'Loss is : {self.loss}') #figure out why the loss is a matrix not a vector, also figure out how to finish


    def predict(self, X):

        self.Z1 = self.W1.dot(self.X) + self.B1
        self.A1 = self.sigmoid(self.Z1)
        self.Z2 = self.W2.dot(self.A1)
        self.A2 = self.sigmoid(self.Z2) + self.B2
        return self.A2

    def save(self):
        pass



         




NN = Neural_Network(2, 1, np.array([[1, 1],[1, 0],[0, 1],[0, 0]]).T, np.array([0, 1, 1, 0]), 0.01)

NN.learn(1000)