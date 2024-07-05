import numpy as np
import matplotlib.pyplot as plt
class Layer: #base layer
    def __init__(self):
        self.input=None
        self.output=None
    def forward(self,input):#forward propagation
        pass
    def backward(self,output_gradient,learning_rate):#backward propagation
        pass

class Dense(Layer):
    def __init__(self,input_size,output_size):
        self.weights=np.random.randn(output_size,input_size) #creates a random 2-D matrix of weights
        self.bias=np.random.randn(output_size,1) #creates a random column vector of biases
    def forward(self,input):
        self.input=input #first layer will take 'THE INPUT' of neural network 
        return np.dot(self.weights,self.input)+self.bias #propagates forward y=wx+b
    def backward(self,output_gradient,learning_rate):
        weights_gradient=np.dot(output_gradient,self.input.T) #}
        self.weights-=learning_rate*weights_gradient #         }  math for calculating input gradient(i.e. change in error w.r.t. input)  
        self.bias-=learning_rate*output_gradient #             }
        return np.dot(self.weights.T,output_gradient)   #change in error w.r.t. input
    
class Activation(Layer): #Layer for passing the output of dense layer through activation function
    def __init__(self,activation,activation_prime):  # parameters are activation function and its derivative
        self.activation=activation
        self.activation_prime=activation_prime
    def forward(self,input):
        self.input=input
        return self.activation(self.input)  #returns the value after passing it in activation function
    def backward(self,output_gradient,learning_rate):
        return np.multiply(output_gradient,self.activation_prime(self.input))  #math for computing change in error w.r.t. input(i.e. output of dense layer)
    
class Tanh(Activation):  #hyperbolic tangent is the activation function as it is bounded inside -1 to 1. we have taken its inheritance from activation class....it acts as a layer only.
    def __init__(self):
        tanh = lambda x:np.tanh(x)              #   } used numpy for tanh function
        tanh_prime= lambda x: 1 - np.tanh(x)**2 #   }
        super().__init__(tanh,tanh_prime)       # constructor telling us how to initialise this class.

def mse(y_true,y_pred):   #cost function mean squared error
    return np.mean(np.power(y_true-y_pred,2)) 
def mse_prime(y_true,y_pred):           # its differentiate
    return 2*(y_pred-y_true)/np.size(y_true)

def predict(network, input):    # function for sequentially propagating the flow of output/input
    output = input
    for layer in network:
        output = layer.forward(output)
    return output

def train(network, loss, loss_prime, x_train, y_train, epochs = 1000, learning_rate = 0.01, verbose = True):
    for e in range(epochs):
        error = 0
        for x, y in zip(x_train, y_train):
            # forward
            output = predict(network, x)

            # error
            error += loss(y, output)

            # backward
            grad = loss_prime(y, output)
            for layer in reversed(network):
                grad = layer.backward(grad, learning_rate)

        error /= len(x_train)     #tells average error
        if verbose:
            print(f"iteration number is : {e + 1}/{epochs}, and error = {error}")


X = np.reshape([[0, 0], [0, 1], [1, 0], [1, 1]], (4, 2, 1)) # for XOR 
Y = np.reshape([[0], [1], [1], [0]], (4, 1, 1))

network = [
    Dense(2, 3),
    Tanh(),
    Dense(3, 1),
    Tanh()
]

# train
train(network, mse, mse_prime, X, Y, epochs=1000, learning_rate=0.1)

for x, y in zip(X, Y):
    output = predict(network, x)
    print(f"Input: {x.T}, Predicted: {output.T}, Actual: {y.T}")

# decision boundary plot
points = []
for x in np.linspace(0, 1, 20):
    for y in np.linspace(0, 1, 20):
        z = predict(network, [[x], [y]])
        points.append([x, y, z[0,0]])

points = np.array(points)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 2], cmap="winter")
plt.show()

