import random
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def cost_derivative(output_activations, y):
    return output_activations - y

def normal_transpose(list):
    return [[row[i] for row in list] for i in range(len(list[0]))]

class Backpropagation_network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(sizes[:-1], sizes[1:])]
    
    #Feedforward is used for predicting output 
    def feedforward(self,a):
        for b, w in zip(self.biases, self.weights):
            a = sigmoid(np.dot(w, a) + b) 
        return a
    
    #Backprop is used for finding gradient in order to update weights and biases
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        activation = x
        activation_list = [x]
        z_list = []
        
        # Forward pass
        for b, w in zip(self.biases, self.weights):
            z = np.dot(w, activation) + b
            z_list.append(z)
            activation = sigmoid(z)
            activation_list.append(activation)
        
        # Backward pass
        delta = cost_derivative(activation_list[-1], y) * sigmoid_prime(z_list[-1]) 
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activation_list[-2].transpose())
        
        for l in range(2, self.num_layers):
            z = z_list[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activation_list[-l-1].transpose())
        
        return (nabla_b, nabla_w)

    #update weights and biases
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x , y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [w - (eta / len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)] # weight - (learning rate/number of batch)*nabla_weight
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)] # bias - (learning rate/number of batch)*nabla_bias

    #Find the sum of the highest values
    def evaluate(self, test):
        test_results = [(np.argmax(self.feedforward(x)), y) for (x, y) in test]
        return sum(int(x == y) for (x, y) in test_results)
    
    #Stoichastic Gradient Descend
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        if test_data:
            n_test = len(test_data)
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print("Epoch {0}: {1} / {2}".format(j, self.evaluate(test_data), n_test))
            else:
                print("Epoch {0} complete".format(j))


from keras.utils import to_categorical
from keras.datasets import mnist

# Load and preprocess the MNIST dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the input data
x_train = [x.reshape(784, 1) / 255.0 for x in x_train]
x_test = [x.reshape(784, 1) / 255.0 for x in x_test]

# Convert labels to one-hot encoding
y_train = [to_categorical(y, 10).reshape(10, 1) for y in y_train]
y_test = [(x, y) for x, y in zip(x_test, y_test)]

# Create training data as list of tuples (x, y)
training_data = list(zip(x_train, y_train))

# Create the neural network
net = Backpropagation_network([784, 30, 10])

# Train the neural network using SGD
net.SGD(training_data, epochs=30, mini_batch_size=10, eta=3.5, test_data=y_test)