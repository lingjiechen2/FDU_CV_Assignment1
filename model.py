import numpy as np
from loss import cross_entropy_loss, delta_cross_entropy_softmax

def relu(Z):
    return np.maximum(0, Z)

def d_relu(Z):
    return Z > 0

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def d_sigmoid(Z):
    s = sigmoid(Z)
    return s * (1 - s)

def tanh(Z):
    return np.tanh(Z)

def d_tanh(Z):
    return 1 - np.tanh(Z)**2

class ThreeLayerNN:
    def __init__(self, input_size, hidden_sizes, output_size, activation='relu', reg_lambda=0.01):
        self.activation_str = activation
        assert activation in ['relu', 'sigmoid', 'tanh'], 'Activation function not supported, please choose from: relu, sigmoid, tanh'
        self.activations = {
            'relu': (relu, d_relu),
            'sigmoid': (sigmoid, d_sigmoid),
            'tanh': (tanh, d_tanh)
        }
        self.activation, self.d_activation = self.activations[activation]
        self.reg_lambda = reg_lambda  # Regularization strength

        self.params = {
            'W1': np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(1. / input_size),
            'b1': np.zeros((1, hidden_sizes[0])),
            'W2': np.random.randn(hidden_sizes[0], hidden_sizes[1]) * np.sqrt(1. / hidden_sizes[0]),
            'b2': np.zeros((1, hidden_sizes[1])),
            'W3': np.random.randn(hidden_sizes[1], output_size) * np.sqrt(1. / hidden_sizes[1]),
            'b3': np.zeros((1, output_size))
        }

    def forward(self, X):
        # Forward pass through first layer
        Z1 = X.dot(self.params['W1']) + self.params['b1']
        A1 = self.activation(Z1)
        
        # Forward pass through second layer
        Z2 = A1.dot(self.params['W2']) + self.params['b2']
        A2 = self.activation(Z2)
        
        # Forward pass through output layer
        Z3 = A2.dot(self.params['W3']) + self.params['b3']
        output = Z3  # No activation function here, softmax will be applied in loss calculation
        
        cache = {'Z1': Z1, 'A1': A1, 'Z2': Z2, 'A2': A2, 'Z3': Z3}
        return output, cache

    def backward(self, X, Y, cache):
        grads = {}
        m = X.shape[0]
        
        # Backpropagation through output layer
        dZ3 = delta_cross_entropy_softmax(cache['Z3'], Y)
        grads['W3'] = (cache['A2'].T.dot(dZ3) + self.reg_lambda * self.params['W3']) / m
        grads['b3'] = np.sum(dZ3, axis=0, keepdims=True) / m
        
        # Backpropagation through second hidden layer
        dA2 = dZ3.dot(self.params['W3'].T)
        dZ2 = dA2 * self.d_activation(cache['Z2'])
        grads['W2'] = (cache['A1'].T.dot(dZ2) + self.reg_lambda * self.params['W2']) / m
        grads['b2'] = np.sum(dZ2, axis=0, keepdims=True) / m
        
        # Backpropagation through first hidden layer
        dA1 = dZ2.dot(self.params['W2'].T)
        dZ1 = dA1 * self.d_activation(cache['Z1'])
        grads['W1'] = (X.T.dot(dZ1) + self.reg_lambda * self.params['W1']) / m
        grads['b1'] = np.sum(dZ1, axis=0, keepdims=True) / m
        
        return grads

    def compute_loss(self, Y_pred, Y_true):
        base_loss = cross_entropy_loss(Y_pred, Y_true)
        reg_loss = 0.5 * self.reg_lambda * (np.sum(np.square(self.params['W1'])) +
                                            np.sum(np.square(self.params['W2'])) +
                                            np.sum(np.square(self.params['W3'])))
        # print(reg_loss)
        return base_loss + reg_loss


