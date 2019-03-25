import numpy as np


def l2_regularization(W, reg_strength):
    """
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    """
    # TODO: Copy from the previous assignment
    loss = reg_strength * np.sum(np.square(W))
    grad = 2.0 * W * reg_strength
    return loss, grad


def softmax_with_cross_entropy(predictions, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient

    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)

    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    N = predictions.shape[0]
    shift_pred = predictions - np.max(predictions, axis=-1)[:, np.newaxis]
    denom = np.sum(np.exp(shift_pred), axis=-1)[:, np.newaxis]
    probs = np.exp(shift_pred) / denom
    target_idx_mat = np.reshape(np.eye(probs.shape[-1])[target_index], predictions.shape)
    loss = -np.sum(target_idx_mat * np.log(probs)) / N
    dprediction = (probs - target_idx_mat) / N

    return loss, dprediction


class Param:
    """
    Trainable parameter of the model
    Captures both parameter value and the gradient
    """

    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)


class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO: Implement forward pass
        # Hint: you'll need to save some information about X
        # to use it later in the backward pass
        self.pos = (X >= 0.0).astype(np.float)
        return np.maximum(X, np.zeros(X.shape))

    def backward(self, d_out):
        """
        Backward pass

        Arguments:
        d_out, np array (batch_size, num_features) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, num_features) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Your final implementation shouldn't have any loops
        d_result = self.pos * d_out
        return d_result

    def params(self):
        # ReLU Doesn't have any parameters
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO: Implement forward pass
        # Your final implementation shouldn't have any loops
        self.X = X.copy()
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        """
        Backward pass
        Computes gradient with respect to input and
        accumulates gradients within self.W and self.B

        Arguments:
        d_out, np array (batch_size, n_output) - gradient
           of loss function with respect to output

        Returns:
        d_result: np array (batch_size, n_input) - gradient
          with respect to input
        """
        # TODO: Implement backward pass
        # Compute both gradient with respect to input
        # and gradients with respect to W and B
        # Add gradients of W and B to their `grad` attribute

        # It should be pretty similar to linear classifier from
        # the previous assignment

        #mean_grad = np.mean(d_out, axis=0)
        # w     (i x o)
        # x     (b x i)
        # b     (1 x o)
        # dout  (b x o)
        # res   (b x i)
        # wg    (i x o)
        # bg    (1 x o)
        self.W.grad += np.dot(self.X.T, d_out)
        self.B.grad += np.sum(d_out, axis=0).reshape(1, d_out.shape[1])
        d_result = np.dot(d_out, self.W.value.T)

        return d_result

    def params(self):
        return {'W': self.W, 'B': self.B}
