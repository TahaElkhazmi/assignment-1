import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from EDF2 import *

# Load the dataset
mnist = datasets.load_digits()
X, y = mnist['data'], mnist['target']
X = X / 16.0  # Normalize the data

# Convert y to one-hot encoding
def onehotmatconverter(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

num_classes = 10  # Number of classes (0-9)
y_one_hot = onehotmatconverter(y, num_classes)

# Split the data into training and test sets using NumPy
train_size = int(0.6 * X.shape[0])  # 60% for training
X_train, X_test = X[:train_size], X[train_size:]
y_train_one_hot, y_test_one_hot = y_one_hot[:train_size], y_one_hot[train_size:]

# MLP network parameters
np.random.seed(42)
batch_size = 10
epochs = 200
learning_rate = 0.02

# Define the MLP
input_mlp = X.shape[1]  # Number of features in the input
hide = 64 # Number of neurons in hidden layers
out = 10       # Number of output classes

# Create computational nodes
x_node = Input()
y_node = Input()

# First hidden layer
h1 = Linear(x_node, input_mlp, hide)
a1 = Sigmoid(h1)

# Second hidden layer
h2 = Linear(a1, hide, hide)
a2 = Sigmoid(h2)

# Output layer
h3 = Linear(a2, hide, out)
output = Softmax(h3)

# Loss function
loss = CrossEntropy(y_node, output)

# Topological sorting
def topological_sort(node):
    visited = set()
    sorted_nodes = []

    def visit(n):
        if n not in visited:
            visited.add(n)
            for input_node in n.inputs:
                visit(input_node)
            sorted_nodes.append(n)

    visit(node)
    return sorted_nodes

# Define the computation graph
graph = topological_sort(loss)
trainables = []
for n in graph:
    if isinstance(n , Parameter):
        trainables.append(n)

# Helper functions
def forward_pass(graph):
    for node in graph:
        node.forward()

def backward_pass(graph):
    for node in reversed(graph):
        node.backward()

def sgd_update(trainables, learning_rate):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

# Train the MLP
for epoch in range(epochs):
    loss_value = 0
    for i in range(0, X_train.shape[0], batch_size):
        x_batch = X_train[i:i + batch_size]
        y_batch = y_train_one_hot[i:i + batch_size]

        x_node.value = x_batch
        y_node.value = y_batch

        # Perform forward and backward passes and update weights
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, learning_rate)

        # Accumulate loss
        loss_value += loss.value

    # Print loss every 1 epochs
    if (epoch + 1) % 1 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

# Evaluate the model's performance
correct_predictions = 0
for i in range(0, X_test.shape[0], batch_size):
    x_batch = X_test[i:i + batch_size]
    y_batch = y_test_one_hot[i:i + batch_size]

    x_node.value = x_batch
    y_node.value = y_batch

    forward_pass(graph)

    # Predict class labels
    predictions = np.argmax(output.value, axis=1)
    correct_predictions += np.sum(predictions == np.argmax(y_batch, axis=1))

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")
