from keras.datasets import mnist
from matplotlib import pyplot
import numpy as np
from EDF3 import *

# 1) Load the MNIST dataset from Keras
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 2) Print the shapes of training and testing data
print('X_train:', train_images.shape)
print('Y_train:', train_labels.shape)
print('X_test: ', test_images.shape)
print('Y_test: ', test_labels.shape)

# 3) Display the first 9 samples from the training set
for display_index in range(9):
    pyplot.subplot(330 + 1 + display_index)
    pyplot.imshow(train_images[display_index], cmap=pyplot.get_cmap('gray'))
pyplot.show()

# 4) Normalize and reshape the data to (N, H, W, Channels)
train_images = train_images.reshape(-1, 28, 28, 1).astype('float32')
test_images = test_images.reshape(-1, 28, 28, 1).astype('float32')

# 5) Define training hyperparameters
NUM_CLASSES = 10      # number of output classes
NUM_EPOCHS = 1        # how many epochs to train
LEARNING_RATE = 0.001 # SGD learning rate
BATCH_SIZE = 64       # mini-batch size

# 6) Create input/label Nodes for the computational graph
input_node = Input()
input_node.value = np.zeros((BATCH_SIZE, 28, 28, 1))  # placeholder shape
label_node = Input()

# 7) Build the CNN layers

# First block: Convolution -> ReLU -> MaxPooling
conv_layer1 = Conv(input_node, in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1)
relu_layer1 = ReLU(conv_layer1)
pool_layer1 = MaxPooling(relu_layer1)

# Second block: Convolution -> ReLU -> MaxPooling
conv_layer2 = Conv(pool_layer1, in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
relu_layer2 = ReLU(conv_layer2)
pool_layer2 = MaxPooling(relu_layer2)

# Third block: Convolution -> ReLU -> MaxPooling
conv_layer3 = Conv(pool_layer2, in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
relu_layer3 = ReLU(conv_layer3)
pool_layer3 = MaxPooling(relu_layer3)

# Fourth block: Convolution -> ReLU -> MaxPooling
conv_layer4 = Conv(pool_layer3, in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
relu_layer4 = ReLU(conv_layer4)
pool_layer4 = MaxPooling(relu_layer4)

# 8) Flatten the output to feed into a Linear layer
flatten_layer = Flatten(pool_layer4)

# 9) Fully-connected (linear) layer to produce logits for 10 classes
fc_layer = Linear(flatten_layer, 10, 128)

# 10) Softmax activation to get class probabilities
predictions_node = Softmax(fc_layer)

# 11) Cross-Entropy loss node to compare predictions vs. true labels
ce_loss = CrossEntropy(label_node, predictions_node)

# 12) Prepare lists to store graph nodes and trainable parameters
node_graph = []
trainable_params = []
epoch_loss_log = []

# ---------------------------------------------------
# 13) Function to gather nodes for the graph and track trainable parameters
def build_graph_and_trainables(last_node, graph_list, trainables_list):
    """
    Recursively traverse the computation graph starting from 'last_node',
    mark each node as visited, and collect trainable parameters.
    """
    graph_list.append(last_node)
    for node_item in graph_list:
        for node_input in node_item.inputs:
            if not node_input.visited:
                graph_list.append(node_input)
                node_input.visited = True
            if node_input.trainable:
                trainables_list.append(node_input)

# ---------------------------------------------------
# 14) Forward pass: evaluate all nodes in reverse topological order
def forward_eval(graph_list):
    """
    Evaluate the .forward() method of each Node in reverse order,
    so that inputs are computed before the node that depends on them.
    """
    for node_item in graph_list[::-1]:
        node_item.forward()

# ---------------------------------------------------
# 15) Backward pass: compute gradients via backprop
def backward_eval(graph_list):
    """
    Call the .backward() method on each Node in forward (topological) order,
    accumulating gradients in each Node.
    """
    for node_item in graph_list:
        node_item.backward()

# ---------------------------------------------------
# 16) SGD update rule
def sgd_optimizer(trainables_list, lr):
    """
    Update trainable parameters by subtracting lr * gradient.
    Special handling for convolution parameters stored in _w and _b.
    """
    for param in trainables_list:
        if not isinstance(param, Conv):
            param.value -= lr * param.gradients[param].T
        else:
            param._w -= lr * param.gradients['weights']
            param._b -= lr * param.gradients['bias']

# ---------------------------------------------------
# 17) Build the graph and list of trainable parameters
build_graph_and_trainables(ce_loss, node_graph, trainable_params)

# 18) Training loop
for epoch_index in range(NUM_EPOCHS):
    total_loss = 0
    # Go through the entire training set in batches
    for batch_start in range(0, train_images.shape[0], BATCH_SIZE):
        batch_end = batch_start + BATCH_SIZE
        input_node.value = train_images[batch_start:batch_end]
        label_node.value = train_labels[batch_start:batch_end]

        # 18a) Forward + Backward + Update
        forward_eval(node_graph)
        backward_eval(node_graph)
        sgd_optimizer(trainable_params, LEARNING_RATE)

        # 18b) Accumulate loss for average tracking
        total_loss += ce_loss.value
        average_loss = total_loss / (train_images.shape[0] / BATCH_SIZE)
        print(ce_loss.value)
        epoch_loss_log.append(average_loss)

    # Print epoch-wise average loss
    print(f"Epoch {epoch_index + 1}, Average Loss: {average_loss:.6f}")

# ---------------------------------------------------
# 19) Evaluate model on the test set
correct_count = 0
for batch_start in range(0, test_images.shape[0], BATCH_SIZE):
    batch_end = batch_start + BATCH_SIZE
    x_batch = test_images[batch_start:batch_end]
    y_batch = test_labels[batch_start:batch_end]

    input_node.value = x_batch
    label_node.value = y_batch

    # Forward pass to get predictions
    forward_eval(node_graph)

    # Compare predicted classes vs. ground truth
    for sample_index in range(len(x_batch)):
        predicted_class = np.argmax(predictions_node.value[sample_index])
        true_class = y_batch[sample_index]
        if predicted_class == true_class:
            correct_count += 1

# 20) Compute and display classification accuracy
accuracy = correct_count / test_images.shape[0]
print(f"Accuracy: {accuracy * 100:.02f}%")
