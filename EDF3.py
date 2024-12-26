import numpy as np

# ---------------------------------------------------
# Base Node class: foundation for all layers/nodes
# ---------------------------------------------------
class Node:
    def __init__(self, inputs=None):
        if inputs is None:
            inputs = []
        self.inputs = inputs        # Incoming nodes
        self.outputs = []           # Outgoing nodes
        self.value = None           # Output value of this node
        self.trainable = False      # Indicates if node has learnable params
        self.visited = False        # For graph-building
        self.gradients = {}         # Gradients w.r.t. this node's inputs

        # Link this node as an output to its input nodes
        for inp_node in inputs:
            inp_node.outputs.append(self)

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError


# ---------------------------------------------------
# Input Node: placeholders for data/labels
# ---------------------------------------------------
class Input(Node):
    def __init__(self):
        super().__init__(inputs=[])

    def forward(self, value=None):
        # If a new value is provided, override self.value
        if value is not None:
            self.value = value

    def backward(self):
        # The gradient of an Input node is typically zero or passed through
        self.gradients = {self: 0}
        for out_node in self.outputs:
            self.gradients[self] += out_node.gradients[self]


# ---------------------------------------------------
# Parameter Node: for storing trainable weights
# ---------------------------------------------------
class Parameter(Node):
    def __init__(self, value):
        super().__init__(inputs=[])
        self.value = value
        self.trainable = True

    def forward(self):
        # Parameters just hold their values
        pass

    def backward(self):
        # Accumulate gradients passed from the next nodes
        self.gradients = {self: 0}
        for out_node in self.outputs:
            self.gradients[self] += out_node.gradients[self]


# ---------------------------------------------------
# Linear layer: fully connected transformation
# ---------------------------------------------------
class Linear(Node):
    def __init__(self, x,output_size,input_size ):
        a = Parameter(np.random.randn(output_size, input_size) * 0.1)
        b = Parameter(np.zeros(output_size))
        Node.__init__(self, [x, a, b])

    def forward(self):
        x, a, b = self.inputs
        self.value = np.dot(x.value, a.value.T) + b.value

    def backward(self):
        x, a, b = self.inputs
        self.gradients[x] = np.dot(self.outputs[0].gradients[self], a.value)
        self.gradients[a] = np.dot(x.value.T, self.outputs[0].gradients[self])
        self.gradients[b] = np.sum(self.outputs[0].gradients[self], axis=0)

# ---------------------------------------------------
# Sigmoid activation layer
# ---------------------------------------------------
class Sigmoid(Node):
    def __init__(self, input_node):
        super().__init__([input_node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        input_value = self.inputs[0].value
        self.value = self._sigmoid(input_value)

    def backward(self):
        grad_from_next = self.outputs[0].gradients[self]
        sigmoid_output = self.value
        # derivative of sigmoid: s*(1-s)
        self.gradients[self.inputs[0]] = grad_from_next * sigmoid_output * (1 - sigmoid_output)


# ---------------------------------------------------
# ReLU activation layer
# ---------------------------------------------------
class ReLU(Node):
    def __init__(self, input_node):
        super().__init__([input_node])

    def forward(self):
        input_data = self.inputs[0]
        # ReLU = max(0, x)
        self.value = np.maximum(0, input_data.value)

    def backward(self):
        input_data = self.inputs[0]
        grad_from_next = self.outputs[0].gradients[self]
        # derivative is 1 for x>0, else 0
        self.gradients[input_data] = grad_from_next * (input_data.value > 0).astype(float)


# ---------------------------------------------------
# Binary Cross Entropy (unused in multi-class, but kept for reference)
# ---------------------------------------------------
class BCE(Node):
    def __init__(self, y_true_node, y_pred_node):
        super().__init__([y_true_node, y_pred_node])

    def forward(self):
        y_true_node, y_pred_node = self.inputs
        epsilon = 1e-12
        clipped_preds = np.clip(y_pred_node.value, epsilon, 1 - epsilon)
        self.value = -np.mean(y_true_node.value * np.log(clipped_preds)
                              + (1 - y_true_node.value) * np.log(1 - clipped_preds))

    def backward(self):
        y_true_node, y_pred_node = self.inputs
        batch_size = y_true_node.value.shape[0]
        epsilon = 1e-12
        clipped_preds = np.clip(y_pred_node.value, epsilon, 1 - epsilon)

        # derivative w.r.t. y_pred
        self.gradients[y_pred_node] = (1.0 / batch_size) * (clipped_preds - y_true_node.value) \
                                      / (clipped_preds * (1 - clipped_preds))
        # derivative w.r.t. y_true is not typically used, so we can store something or zero
        self.gradients[y_true_node] = (1.0 / batch_size) * (np.log(clipped_preds) - np.log(1 - clipped_preds))


# ---------------------------------------------------
# Softmax activation layer (multi-class)
# ---------------------------------------------------
class Softmax(Node):
    def __init__(self, input_node):
        super().__init__([input_node])

    def forward(self):
        input_value = self.inputs[0].value
        shifted_input = input_value - np.max(input_value, axis=1, keepdims=True)  # stability
        exp_values = np.exp(shifted_input)
        self.value = exp_values / np.sum(exp_values, axis=1, keepdims=True)

    def backward(self):
        # For multi-class tasks, cross-entropy typically handles the gradient
        # This pass-through is simplified; the cross-entropy node will do the main logic
        grad_from_next = self.outputs[0].gradients[self]
        self.gradients[self.inputs[0]] = grad_from_next


# ---------------------------------------------------
# CrossEntropy: multi-class classification
# ---------------------------------------------------
class CrossEntropy(Node):
    def __init__(self, y_true_node, y_pred_node):
        super().__init__([y_true_node, y_pred_node])

    def forward(self):
        y_true = self.inputs[0].value
        y_pred = self.inputs[1].value
        epsilon = 1e-12
        clipped_preds = np.clip(y_pred, epsilon, 1. - epsilon)
        # Negative log-likelihood of the correct class
        self.value = -np.mean(np.log(clipped_preds[np.arange(len(y_true)), y_true]))

    def backward(self):
        y_true = self.inputs[0].value
        y_pred = self.inputs[1].value
        epsilon = 1e-12
        clipped_preds = np.clip(y_pred, epsilon, 1. - epsilon)

        # Gradient wrt y_pred: subtract 1 in the correct class location
        grad_y_pred = clipped_preds.copy()
        grad_y_pred[np.arange(len(y_true)), y_true] -= 1
        grad_y_pred /= len(y_true)

        # Cross-entropy doesn't backprop w.r.t. y_true in typical usage
        self.gradients[self.inputs[0]] = np.zeros_like(y_true)
        self.gradients[self.inputs[1]] = grad_y_pred


# ---------------------------------------------------
# Conv (Convolution) Node
# ---------------------------------------------------
class Conv(Node):
    def __init__(self, input_node, in_channels, out_channels,
                 kernel_size=3, stride=1, padding=0):
        super().__init__([input_node])
        self.trainable = True
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initialize convolution filters (weights) and bias
        self._w = np.random.normal(loc=0, scale=0.1,
                                   size=(kernel_size, kernel_size, in_channels, out_channels))
        self._b = np.ones((1, 1, 1, out_channels))  # bias for each filter

    def calculate_output_dims(self, input_dims: tuple) -> tuple:
        """
        Compute output dimensions based on input shape, kernel size, stride, padding.
        input_dims = (N, H_in, W_in, C_in)
        Returns (N, H_out, W_out, C_out).
        """
        batch_size, h_in, w_in, c_in = input_dims
        k_h, k_w, _, num_filters = self._w.shape
        stride = self.stride

        h_out = (h_in - k_h + 2 * self.padding) // stride + 1
        w_out = (w_in - k_w + 2 * self.padding) // stride + 1
        return (batch_size, h_out, w_out, num_filters)

    def pad_input(self, input_data):
        """
        Apply zero-padding around the spatial dimensions if padding > 0.
        """
        if self.padding == 0:
            return input_data
        return np.pad(input_data,
                      pad_width=((0, 0),
                                 (self.padding, self.padding),
                                 (self.padding, self.padding),
                                 (0, 0)),
                      mode='constant', constant_values=0)

    def forward(self):
        # Forward pass for convolution
        prev_value = self.inputs[0].value
        prev_value_padded = self.pad_input(prev_value)
        out_shape = self.calculate_output_dims(prev_value.shape)
        n, h_in, w_in, _ = prev_value_padded.shape
        _, h_out, w_out, _ = out_shape
        k_h, k_w, _, num_filters = self._w.shape

        output = np.zeros(out_shape)

        # Slide the kernel across every valid position
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + k_h
                w_end = w_start + k_w

                patch = prev_value_padded[:, h_start:h_end, w_start:w_end, :]
                # sum over (k_h, k_w, in_channels) for each out_channels
                output[:, i, j, :] = np.sum(
                    patch[:, :, :, :, np.newaxis] * self._w[np.newaxis, :, :, :],
                    axis=(1, 2, 3)
                )

        self.value = output + self._b

    def backward(self):
        grad_output = self.outputs[0].gradients[self]
        prev_value = self.inputs[0].value
        prev_value_padded = self.pad_input(prev_value)
        n, h_in, w_in, c_in = prev_value.shape
        k_h, k_w, _, num_filters = self._w.shape
        _, h_out, w_out, _ = grad_output.shape

        # Initialize gradients
        grad_w = np.zeros_like(self._w)
        grad_b = np.sum(grad_output, axis=(0, 1, 2))
        grad_prev_padded = np.zeros_like(prev_value_padded)

        # Compute gradients w.r.t. weights and input
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + k_h
                w_end = w_start + k_w

                patch = prev_value_padded[:, h_start:h_end, w_start:w_end, :]

                grad_w += np.sum(
                    patch[:, :, :, :, np.newaxis] * grad_output[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=0
                )

                grad_prev_padded[:, h_start:h_end, w_start:w_end, :] += np.sum(
                    self._w[np.newaxis, :, :, :, :] *
                    grad_output[:, i:i+1, j:j+1, np.newaxis, :],
                    axis=4
                )

        # Remove padding from gradient if needed
        if self.padding > 0:
            grad_prev = grad_prev_padded[:, self.padding:-self.padding, self.padding:-self.padding, :]
        else:
            grad_prev = grad_prev_padded

        # Store computed gradients
        self.gradients['weights'] = grad_w
        self.gradients['bias'] = grad_b
        self.gradients[self.inputs[0]] = grad_prev


# ---------------------------------------------------
# Max Pooling Node
# ---------------------------------------------------
class MaxPooling(Node):
    def __init__(self, input_node, kernel_size=2, stride=2):
        super().__init__([input_node])
        self.kernel_size = kernel_size
        self.stride = stride
        self._mask = {}

    def forward(self):
        prev_value = self.inputs[0].value
        self._shape = prev_value.shape
        n, h_in, w_in, c = prev_value.shape
        h_pool, w_pool = self.kernel_size, self.kernel_size
        h_out = 1 + (h_in - h_pool) // self.stride
        w_out = 1 + (w_in - w_pool) // self.stride
        output = np.zeros((n, h_out, w_out, c))

        # Slide the pooling window
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + h_pool
                w_end = w_start + w_pool
                patch = prev_value[:, h_start:h_end, w_start:w_end, :]
                # Identify the max in each patch
                self._mask[(i, j)] = (patch == np.max(patch, axis=(1, 2), keepdims=True))
                output[:, i, j, :] = np.max(patch, axis=(1, 2))

        self.value = output

    def backward(self):
        grad_output = self.outputs[0].gradients[self]
        n, h_in, w_in, c = self._shape
        h_pool, w_pool = self.kernel_size, self.kernel_size
        d_prev = np.zeros(self._shape)

        h_out, w_out, _ = grad_output.shape[1:]

        # Pass gradients back only to max positions
        for i in range(h_out):
            for j in range(w_out):
                h_start = i * self.stride
                w_start = j * self.stride
                h_end = h_start + h_pool
                w_end = w_start + w_pool

                mask_region = self._mask[(i, j)]
                d_prev[:, h_start:h_end, w_start:w_end, :] += mask_region * grad_output[:, i:i+1, j:j+1, :]

        self.gradients[self.inputs[0]] = d_prev


# ---------------------------------------------------
# Flatten Node: reshapes (N, H, W, C) -> (N, H*W*C)
# ---------------------------------------------------
class Flatten(Node):
    def __init__(self, input_node):
        super().__init__([input_node])
        self.input_shape = None

    def forward(self):
        input_value = self.inputs[0].value
        self.input_shape = input_value.shape
        # Flatten except the batch dimension
        self.value = input_value.reshape(input_value.shape[0], -1)

    def backward(self):
        grad_from_next = self.outputs[0].gradients[self]
        self.gradients[self.inputs[0]] = grad_from_next.reshape(self.input_shape)
