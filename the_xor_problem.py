import numpy as np
import matplotlib.pyplot as plt
from EDF import *

# Step 1: Generate the XOR Dataset
np.random.seed(42)
n_samples = 200  # Total number of samples
n_per_class = n_samples // 2

# Class 0
mean_00, mean_01 = [-1, -1], [1, 1]
cov = [[0.1, 0], [0, 0.1]]  # Covariance matrix

X_00 = np.random.multivariate_normal(mean_00, cov, n_per_class // 2)
X_01 = np.random.multivariate_normal(mean_01, cov, n_per_class // 2)

# Class 1
mean_10, mean_11 = [-1, 1], [1, -1]
X_10 = np.random.multivariate_normal(mean_10, cov, n_per_class // 2)
X_11 = np.random.multivariate_normal(mean_11, cov, n_per_class // 2)

# Combine samples
X = np.vstack([X_00, X_01, X_10, X_11])
y = np.hstack([np.zeros(n_per_class), np.ones(n_per_class)])

# Step 2: Plot the XOR Dataset
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("XOR Dataset")
plt.show()

# Step 3: Train the Logistic Regression Model
# Split the data into training and testing sets
test_size = 0.25
indices = np.arange(X.shape[0])
np.random.shuffle(indices)
split = int(len(X) * (1 - test_size))

X_train, X_test = X[indices[:split]], X[indices[split:]]
y_train, y_test = y[indices[:split]], y[indices[split:]]

# Initialize nodes
x_node = Input()
y_node = Input()
w_node = Parameter(np.random.randn(X_train.shape[1], 1) * 0.1)  # Weights
b_node = Parameter(np.zeros(1))  # Bias

# Build the computation graph
u_node = Linear(x_node, w_node, b_node)
sigmoid = Sigmoid(u_node)
loss = BCE(y_node, sigmoid)
graph = [x_node, y_node, w_node, b_node, u_node, sigmoid, loss]
trainable = [w_node, b_node]

# Training loop
epochs = 100
batch_size = 10
learning_rate = 0.01

def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in reversed(graph):
        n.backward()

def sgd_update(trainables, learning_rate):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]

for epoch in range(epochs):
    loss_value = 0
    for i in range(0, X_train.shape[0], batch_size):
        x_node.value = X_train[i:i+batch_size]
        y_node.value = y_train[i:i+batch_size].reshape(-1, 1)

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, learning_rate)

        loss_value += loss.value

    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0]}")

# Step 4: Evaluate the Model
correct_predictions = 0
for i in range(X_test.shape[0]):
    x_node.value = X_test[i:i+1]
    forward_pass(graph)
    prediction = sigmoid.value >= 0.5
    if prediction == y_test[i]:
        correct_predictions += 1

accuracy = correct_predictions / X_test.shape[0]
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = np.zeros(xx.shape)
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x_node.value = np.array([[xx[i, j], yy[i, j]]])
        forward_pass(graph)
        Z[i, j] = sigmoid.value

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Decision Boundary")
plt.show()
