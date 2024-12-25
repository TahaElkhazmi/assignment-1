import numpy as np
import matplotlib.pyplot as pyplot
from keras.datasets import mnist
from EDF2 import *

#loading the dataset
(train_X, train_y), (test_X, test_y) = mnist.load_data()

#printing the shapes of the vectors
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  '  + str(test_X.shape))
print('Y_test:  '  + str(test_y.shape))


for i in range(9):
    pyplot.subplot(330 + 1 + i)
    pyplot.imshow(train_X[i], cmap=pyplot.get_cmap('gray'))
pyplot.show()

# Normalize data
train_X = train_X.reshape((train_X.shape[0], -1)) / 255.0
test_X = test_X.reshape(test_X.shape[0], -1) / 255.0

# Convert labels to one-hot encoding
def onehot_encode(y, num_classes):
    one_hot = np.zeros((y.shape[0], num_classes))
    one_hot[np.arange(y.shape[0]), y] = 1
    return one_hot

num_classes = 10
train_y_one_hot = onehot_encode(train_y, num_classes)
test_y_one_hot = onehot_encode(test_y, num_classes)

# MLP network parameters
np.random.seed(42)
batch_size = 64
epochs = 10
learning_rate = 0.01

# Define the MLP
input_dim = 784  # Number of features in the input
hidden_units = 256  # Number of neurons in hidden layers
output_dim = 10  # Number of output classes

# Create computational nodes
x_node = Input()
y_node = Input()

# First hidden layer
h1 = Linear(x_node, input_dim, hidden_units)
a1 = Sigmoid(h1)

# Second hidden layer
h2 = Linear(a1, hidden_units, hidden_units)
a2 = Sigmoid(h2)

# Output layer
h3 = Linear(a2, hidden_units, output_dim)
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
trainables = [n for n in graph if isinstance(n, Parameter)]

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
loss_history = []
for epoch in range(epochs):
    epoch_loss = 0
    for i in range(0, train_X.shape[0], batch_size):
        x_batch = train_X[i:i + batch_size]
        y_batch = train_y_one_hot[i:i + batch_size]

        x_node.value = x_batch
        y_node.value = y_batch

        # Perform forward and backward passes and update weights
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, learning_rate)

        # Accumulate loss
        epoch_loss += loss.value

    loss_history.append(epoch_loss / train_X.shape[0])
    print(f"Epoch {epoch + 1}, Loss: {epoch_loss / train_X.shape[0]}")


# Evaluate the model's performance
correct_predictions = 0
for i in range(0, test_X.shape[0], batch_size):
    x_batch = test_X[i:i + batch_size]
    y_batch = test_y_one_hot[i:i + batch_size]

    x_node.value = x_batch
    y_node.value = y_batch

    forward_pass(graph)

    # Predict class labels
    predictions = np.argmax(output.value, axis=1)
    correct_predictions += np.sum(predictions == np.argmax(y_batch, axis=1))

accuracy = correct_predictions / test_X.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

# Visualize some predictions (Chat gpt <-----------)
pyplot.figure(figsize=(12, 8))
for i in range(9):
    pyplot.subplot(3, 3, i + 1)
    pyplot.imshow(test_X[i].reshape(28, 28), cmap='gray')
    x_node.value = test_X[i].reshape(1, -1)
    forward_pass(graph)
    predicted_label = np.argmax(output.value, axis=1)[0]
    true_label = test_y[i]
    pyplot.title(f"Pred: {predicted_label}, True: {true_label}")
    pyplot.axis('off')
pyplot.tight_layout()
pyplot.show()