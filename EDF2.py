import numpy as np

# Base Node class
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs
        self.outputs = []
        self.value = None
        self.gradients = {}

        for node in inputs:
            node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# Input Node
class Input(Node):
    def __init__(self):
        Node.__init__(self)

    def forward(self, value=None):
        if value is not None:
            self.value = value

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]


# Parameter Node
class Parameter(Node):
    def __init__(self, value):
        Node.__init__(self)
        self.value = value

    def forward(self):
        pass

    def backward(self):
        self.gradients = {self: 0}
        for n in self.outputs:
            self.gradients[self] += n.gradients[self]

class Multiply(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise multiplication
        x, y = self.inputs
        self.value = x.value * y.value

    def backward(self):
        # Compute gradients for x and y based on the chain rule
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self] * y.value
        self.gradients[y] = self.outputs[0].gradients[self] * x.value

class Addition(Node):
    def __init__(self, x, y):
        # Initialize with two inputs x and y
        Node.__init__(self, [x, y])

    def forward(self):
        # Perform element-wise addition
        x, y = self.inputs
        self.value = x.value + y.value

    def backward(self):
        # The gradient of addition with respect to both inputs is the gradient of the output
        x, y = self.inputs
        self.gradients[x] = self.outputs[0].gradients[self]
        self.gradients[y] = self.outputs[0].gradients[self]


# Sigmoid Activation Node
class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        partial = self.value * (1 - self.value)
        self.gradients[self.inputs[0]] = partial * self.outputs[0].gradients[self]

class BCE(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true, y_pred = self.inputs
        self.value = np.sum(-y_true.value*np.log(y_pred.value)-(1-y_true.value)*np.log(1-y_pred.value))

    def backward(self):
        epsilon = 1e-8
        y_true, y_pred = self.inputs
        self.gradients[y_pred] = (1 / (y_true.value.shape[0]+epsilon)) * (y_pred.value - y_true.value)/(y_pred.value*(1-y_pred.value))
        self.gradients[y_true] = (1 / (y_true.value.shape[0]+epsilon)) * (np.log(y_pred.value) - np.log(1-y_pred.value))


class Linear(Node):
    def __init__(self,x , num1 , num2):
        w1 = Parameter(np.random.randn(num1, num2) * 0.5)
        w0 = Parameter(np.zeros(num2))
        Node.__init__(self,[x,w1,w0])



    def forward(self):
        x , w1 , w0 = self.inputs
        self.value = np.dot(x.value , w1.value) +w0.value

    def backward(self):
        x, w1, w0 = self.inputs
        self.gradients[x] = np.dot(self.outputs[0].gradients[self],w1.value.T)
        self.gradients[w1] = np.dot(x.value.T,self.outputs[0].gradients[self])
        self.gradients[w0] = np.sum(self.outputs[0].gradients[self],axis=0)

class Softmax(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._softmax(input_value)

    def backward(self):

        y_true = self.outputs[0].inputs[0].value  # Get true labels from CrossEntropy
        self.gradients[self.inputs[0]] = (self.value - y_true) / y_true.shape[0]


class CrossEntropy(Node):
    def __init__(self, y_true, y_pred):
        Node.__init__(self, [y_true, y_pred])

    def forward(self):
        y_true = self.inputs[0].value
        y_pred = self.inputs[1].value

        epsilon = 1e-9
        self.value = -np.sum(y_true * np.log(y_pred + epsilon)) / y_true.shape[0]

    def backward(self):
        y_true = self.inputs[0].value
        y_pred = self.inputs[1].value

        self.gradients[self.inputs[0]] = -(np.log(y_pred + 1e-9)) / y_true.shape[0]
        self.gradients[self.inputs[1]] = (y_pred - y_true) / y_true.shape[0]



