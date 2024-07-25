import random
import numpy as np
import sys
import json
import os

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

#return a 10-dimensional unit vector with a 1.0 in yth position and zeroes for the rest
def vectorized_result(y):
    e = np.zeros((10, 1))
    e[int(y)] = 1.0
    return e

class Quadratic_Cost(object):
    @staticmethod
    def fn(a, y):
        return 0.5 * np.linalg.norm(a - y) ** 2
    
    @staticmethod
    def delta(z, a, y):
        return (a - y) * sigmoid_prime(z)
    
class Cross_Entropy_Cost(object):
    @staticmethod
    def fn(a, y):
        return np.sum(np.nan_to_num(-y * np.log(a) - (1 - y) * np.log(1 - a)))
    
    @staticmethod
    def delta(z, a, y):
        return (a - y)

#transpose the normal list, not numpy list   
def normal_transpose(list):
    return [[row[i] for row in list] for i in range(len(list[0]))]

class Cross_entropy_network(object):
    def __init__(self, sizes, cost=Cross_Entropy_Cost):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.advanced_weight_initialization()
        self.cost = cost

    #advanced weight initialization
    def advanced_weight_initialization(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x)/(np.sqrt(x)) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    #old weight initialization
    def old_weight_initialization(self):
        self.biases = [np.random.randn(y, 1) for y in self.sizes[1:]]
        self.weights = [np.random.randn(y,x) for x,y in zip(self.sizes[:-1], self.sizes[1:])]

    #Feedforward is used for predicting output 
    def feedforward(self, a):
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
        delta = (self.cost).delta(z_list[-1], activation_list[-1], y) #delta = a - y
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, normal_transpose(activation_list[-2]))
        
        for l in range(2, self.num_layers):
            z = z_list[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, normal_transpose(activation_list[-l-1]))
        
        return (nabla_b, nabla_w)
    
    #update weights and biases
    def update_mini_batch(self, mini_batch, eta, lmda, n):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x , y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        self.weights = [(1 - (eta * lmda / n)) * w - (eta / len(mini_batch)) * nw 
                        for w, nw in zip(self.weights, nabla_w)] # weight - (learning rate/number of batch)*nabla_weight
        self.biases = [b - (eta / len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)] # bias - (learning rate/number of batch)*nabla_bias

    #convert=False for validation or test data, True for training data
    def evaluate(self, data, convert=False):
        if convert:
            results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for (x, y) in data]
        else:
            results = [(np.argmax(self.feedforward(x)), y) for (x, y) in data]
        return sum(int(x == y) for (x, y) in results)
    
    #Regularization
    #convert=False for training data and True for validation or test data
    def total_cost(self, data, lmda, convert=False):
        cost = 0.0
        for x, y in data:
            a = self.feedforward(x)
            if convert: 
                y = vectorized_result(y)
            cost += self.cost.fn(a, y) / len(data)
        cost += (lmda / (2 * len(data))) * sum(np.linalg.norm(w) ** 2 for w in self.weights)
        return cost
    
    #Stoichastic gradient descent
    def SGD(self, training_data, epochs, mini_batch_size, eta, lmda = 0.0,
            evaluation_data = None, monitor_evaluation_cost = False,
            monitor_evaluation_accuracy = False, monitor_training_cost = False, 
            monitor_training_accuracy = False):
        """ if os.path.isfile(filename):
            net = load(filename)
            self.weights = net.weights
            self.biases = net.biases """
        if evaluation_data:
            n_data = len(evaluation_data)
        n = len(training_data)
        evaluation_cost, evaluation_accuracy = [], []
        training_cost, training_accuracy = [], []
        for i in range(epochs):
            random.shuffle(training_data)
            mini_batches = [
                training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)
            ]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta, lmda, len(training_data))
            print(f"Epoch {i} training complete")
            if monitor_training_cost:
                cost = self.total_cost(training_data, lmda)
                training_cost.append(cost)
                print(f"Cost on training data: {cost}")
            if monitor_training_accuracy:
                accuracy = self.evaluate(training_data, convert=True)
                training_accuracy.append(accuracy)
                print(f"Accuracy on training data: {accuracy} / {n}")
            if monitor_evaluation_cost:
                cost = self.total_cost(evaluation_data, lmda, convert=True)
                evaluation_cost.append(cost)
                print(f"Cost on evaluation data: {cost}")
            if monitor_evaluation_accuracy:
                accuracy = self.evaluate(evaluation_data)
                evaluation_accuracy.append(accuracy)
                print(f"Accuracy on evaluation data: {accuracy} / {n_data}")
            print()
        #self.save(filename)
        return evaluation_cost, evaluation_accuracy, training_cost, training_accuracy
    
    #save network stats
    def save(self, filename):
        data = {"sizes": self.sizes,
                "weights": [w.tolist() for w in self.weights],
                "biases": [b.tolist() for b in self.biases],
                "cost": str(self.cost.__name__)}
        f = open(filename, "w")
        json.dump(data, f)
        f.close()
        
def load(filename):
    f = open(filename, "r")
    data = json.load(f)
    f.close()
    cost = getattr(sys.modules[__name__], data["cost"])
    net = Cross_entropy_network(data["sizes"], cost=cost)
    net.weights = [np.array(w) for w in data["weights"]]
    net.biases = [np.array(b) for b in data["biases"]]
    return net


"""
To run the code:
import mnist_loader
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
import Cross_entropy
net = Cross_entropy.Cross_entropy_network([784, 30, 10], cost=Cross_entropy.Cross_Entropy_Cost)
net.SGD(
    training_data, epochs=30, mini_batch_size=10, eta=0.25, lmda=5.0,
    evaluation_data=validation_data,
    monitor_evaluation_cost=True, monitor_evaluation_accuracy=True,
    monitor_training_cost=True, monitor_training_accuracy=True
)
"""

"""
To find the suitable hyper-parameters:
1. Adjusting lamda:
net = Cross_entropy.Cross_entropy_network([784, 10])
net.SGD(training_data[:1000], 30, 10, 10.0, lmda = 1000.0, evaluation_data = validation_data[:100], monitor_evaluation_accuracy=True)
net.SGD(training_data[:1000], 30, 10, 10.0, lmda = 20.0, evaluation_data = validation_data[:100], monitor_evaluation_accuracy=True)
net.SGD(training_data[:1000], 30, 10, 1.0, lmda = 20.0, evaluation_data = validation_data[:100], monitor_evaluation_accuracy=True)
2. Adjust learning rate

Choose Reasonable Initial Values: Start with values commonly used in practice or based on prior knowledge. For example, a mini-batch size of 32, a learning rate of 0.01, and lambda of 0.1.

Experiment and Tune: Monitor the training and validation performance while adjusting the parameters. Use techniques such as grid search or random search to explore the parameter space efficiently.

Regularization Parameter Selection: If using regularization, consider the complexity of your model and the amount of available data. Cross-validation can help identify an appropriate lambda value.

Adaptive Techniques: Consider using adaptive learning rate algorithms (e.g., Adam, RMSprop) that adjust the learning rate during training based on past gradients or other factors.
"""



