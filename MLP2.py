import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from EDF2 import *


# Generate XOR dataset
N_SAMPLES = 400
MEAN1 = [1, 1]
MEAN2 = [3, 3]
MEAN3 = [3, 1]
MEAN4 = [1, 3]
COV = [[0.1, 0], [0, 0.1]]
TEST_SIZE = 0.25
# Sample data from Gaussian distributions
X1 = multivariate_normal.rvs(mean=MEAN1, cov=COV, size=N_SAMPLES // 4)
X2 = multivariate_normal.rvs(mean=MEAN2, cov=COV, size=N_SAMPLES // 4)
X3 = multivariate_normal.rvs(mean=MEAN3, cov=COV, size=N_SAMPLES // 4)
X4 = multivariate_normal.rvs(mean=MEAN4, cov=COV, size=N_SAMPLES // 4)

# Combine data and assign labels
X = np.vstack([X1, X2, X3, X4])
y = np.hstack([np.zeros(N_SAMPLES // 2), np.ones(N_SAMPLES // 2)])


# Shuffle data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
X, y = X[indices], y[indices]
test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]
X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]
# Visualize the dataset
plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='viridis', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('XOR Dataset')
plt.show()

taha = y
# MLP network parameters
np.random.seed(42)
batch_size = 10
epochs = 400
learning_rate = 0.02

# Define the architecture of the MLP
input_mlp = X.shape[1]  # Number of features in the input
hide = 20         # Number of  hidden layers
out = 1          # Single output

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
output = Sigmoid(h3)

# Define the loss function (Binary Cross-Entropy)
loss = BCE(y_node, output)
def topological_sort(node):

    visited = set()
    sorted_nodes = []

    def visitedloop(n):
        if n not in visited:
            visited.add(n)
            for input_node in n.inputs:
                visitedloop(input_node)
            sorted_nodes.append(n)

    visitedloop(node)
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
    for i in range(0, X.shape[0], batch_size):

        x_batch = X_train[i:i + batch_size]
        y_batch = y_train[i:i + batch_size].reshape(-1, 1)


        x_node.value = x_batch
        y_node.value = y_batch

        # Perform forward and backward passes and update weights
        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainables, learning_rate)

        # Accumulate loss
        loss_value += loss.value

    # Print loss every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}, Loss: {loss_value / X.shape[0]}")

# Evaluate the model's performance
correct_predictions = 0
for i in range(0, X.shape[0], batch_size):
    x_batch = X_test[i:i + batch_size]
    y_batch = y_test[i:i + batch_size].reshape(-1, 1)

    x_node.value = x_batch
    y_node.value = y_batch

    forward_pass(graph)

    # Predict and count correct predictions
    predictions = (output.value >= 0.5).astype(int)
    correct_predictions += np.sum(predictions == y_batch)

accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.2f}%")

# Plot decision boundaries
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

# Compute predictions for each point in the grid
Z = []
for x, y in zip(xx.ravel(), yy.ravel()):
    x_node.value = np.array([[x, y]])
    forward_pass(graph)
    Z.append(output.value.item())

Z = np.array(Z).reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, levels=50, cmap='viridis', alpha=0.8)  # Decision boundary

plt.scatter(X[:, 0], X[:, 1], c=taha, cmap='viridis', edgecolor='k')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary of MLP')
plt.show()


