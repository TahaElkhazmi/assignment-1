from EDF import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Define constants
batchsize = 12
CLASS1_SIZE = 100
CLASS2_SIZE = 100
N_FEATURES = 2
N_OUTPUT = 1
LEARNING_RATE = 0.02
epochs = 100
TEST_SIZE = 0.25

# Define the means and covariances of the two components
MEAN1 = np.array([-1, 0])
COV1 = np.array([[1, 0], [0, 1]])
MEAN2 = np.array([-1, 2])
COV2 = np.array([[1, 0], [0, 1]])

# Generate random points from the two components
X1 = multivariate_normal.rvs(MEAN1, COV1, CLASS1_SIZE)
X2 = multivariate_normal.rvs(MEAN2, COV2, CLASS2_SIZE)

# Combine the points and generate labels
X = np.vstack((X1, X2))
y = np.hstack((np.zeros(CLASS1_SIZE), np.ones(CLASS2_SIZE)))

# Plot the generated data
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Scatter Plot of Generated Data')
plt.show()

# Split data
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

test_set_size = int(len(X) * TEST_SIZE)
test_indices = indices[:test_set_size]
train_indices = indices[test_set_size:]

X_train, X_test = X[train_indices], X[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

# Model parameters
n_features = X_train.shape[1]
n_output = 1

# Initialize weights and biases
W0 = np.zeros(1)
W1 = np.random.randn(1) * 0.1
W2 = np.random.randn(1) * 0.1
W = np.array([W1 , W2])

# Create nodes
x_node = Input()
y_node = Input()

w0_node = Parameter(W0)
w1_node = Parameter(W)



# Build computation graph
u_node = Linear(x_node , w1_node , w0_node)
sigmoid = Sigmoid(u_node)
loss = BCE(y_node, sigmoid)

# Create graph outside the training loop
graph = [x_node,w1_node,w0_node,u_node,sigmoid,loss]
trainable = [w0_node ,w1_node]

# Training loop
#epochs = 100
#learning_rate = 0.001

# Forward and Backward Pass
def forward_pass(graph):
    for n in graph:
        n.forward()

def backward_pass(graph):
    for n in graph[::-1]:
        n.backward()

# SGD Update
def sgd_update(trainables, learning_rate=1e-2):
    for t in trainables:
        t.value -= learning_rate * t.gradients[t]


for epoch in range(epochs):
    loss_value = 0
    for i in range(0 , X_train.shape[0] ,batchsize):
        x_node.value = X_train[i: i+batchsize]
        y_node.value = y_train[i:i+batchsize].reshape(-1, 1)

        forward_pass(graph)
        backward_pass(graph)
        sgd_update(trainable, LEARNING_RATE)

        loss_value += loss.value/batchsize

    print(f"Epoch {epoch + 1}, Loss: {loss_value / X_train.shape[0] / batchsize}")

# Evaluate the model
correct_predictions = 0
for i in range(0 , X_test.shape[0] , batchsize):
    x_batch = X_test[i:i+batchsize]
    y_batch = y_test[i:i+batchsize].reshape(-1 ,1)
    if x_batch.shape[0] == 0:  # Skip empty batches
        continue
    x_node.value = x_batch
    y_node.value = y_batch
    forward_pass(graph)

    predictions = sigmoid.value >= 0.5  # Threshold at 0.5
    correct_predictions += np.sum(predictions == y_batch)




accuracy = correct_predictions / X_test.shape[0]
print(f"Accuracy: {accuracy * 100:.02f}%")

x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max), np.linspace(y_min, y_max))
Z = []
for i,j in zip(xx.ravel(),yy.ravel()):
    x_node.value = np.array([i ,j]).reshape(1, -1)

    forward_pass(graph)

    Z.append(sigmoid.value.item())

Z = np.array(Z).reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary')
plt.show()

batch_sizes = [1, 2, 4, 6, 8]  # Different batch sizes to test
loss_results = {}  # Dictionary to store losses for each batch size

# Training loop for each batch size
for batchsize in batch_sizes:
    # Reinitialize parameters and graph for each batch size
    W0 = np.zeros(1)
    W1 = np.random.randn(1) * 0.1
    W2 = np.random.randn(1) * 0.1
    W = np.array([W1, W2])

    # Create input and parameter nodes
    x_node = Input()
    y_node = Input()
    w0_node = Parameter(W0)
    w1_node = Parameter(W)

    # Build computation graph
    u_node = Linear(x_node, w1_node, w0_node)
    sigmoid = Sigmoid(u_node)
    loss = BCE(y_node, sigmoid)

    graph = [x_node, w1_node, w0_node, u_node, sigmoid, loss]
    trainable = [w0_node, w1_node]


    losses = []  # Store losses for this batch size

    # Training loop for each epoch
    for epoch in range(epochs):
        epoch_loss = 0
        for i in range(0, X_train.shape[0], batchsize):
            # Pepare input and target data for this batch
            x_node.value = X_train[i:i+batchsize]
            y_node.value = y_train[i:i+batchsize].reshape(-1, 1)

            forward_pass(graph)  # Forward pass
            backward_pass(graph)  # Backward pass
            sgd_update(trainable, LEARNING_RATE)  # Update weights

            # Compute the average loss for this batch
            epoch_loss += loss.value / batchsize

        avg_loss = epoch_loss / (X_train.shape[0] // batchsize)
        losses.append(avg_loss)  # Store the average loss for this epoch

    # Store the losses for this batch size in the results dictionary
    loss_results[batchsize] = losses

# Plotting the results
plt.figure(figsize=(10, 6))
for batchsize, losses in loss_results.items():
    plt.plot(range(epochs), losses, label=f'Batch Size: {batchsize}')
plt.xlabel('Epochs')
plt.ylabel('Average Training Loss')
plt.title('Effect of Batch Size on Training Loss')
plt.legend()
plt.show()
