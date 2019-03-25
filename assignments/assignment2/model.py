import numpy as np

from layers import FullyConnectedLayer, ReLULayer, softmax_with_cross_entropy, l2_regularization


class TwoLayerNet:
    """ Neural network with two fully connected layers """

    def __init__(self, n_input, n_output, hidden_layer_size, reg):
        """
        Initializes the neural network

        Arguments:
        hidden_layer_size, int - number of neurons in the hidden layer
        n_input, int - dimension of the model input
        n_output, int - number of classes to predict
        reg, float - L2 regularization strength
        """
        self.reg = reg
        # TODO Create necessary layers
        self.fc1 = FullyConnectedLayer(n_input, hidden_layer_size)
        self.relu1 = ReLULayer()
        self.fc2 = FullyConnectedLayer(hidden_layer_size, n_output)

    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples

        Arguments:
        X, np array (batch_size, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        # TODO Set parameter gradient to zeros
        # Hint: using self.params() might be useful!
        for k, param in self.params().items():
            param.grad[:] = 0.0

        # TODO Compute loss and fill param gradients
        # by running forward and backward passes through the model
        res = X
        res = self.fc1.forward(res)
        res = self.relu1.forward(res)
        res = self.fc2.forward(res)
        loss, grad = softmax_with_cross_entropy(res, y)
        grad = self.fc2.backward(grad)
        grad = self.relu1.backward(grad)
        grad = self.fc1.backward(grad)

        # After that, implement l2 regularization on all params
        # Hint: self.params() is useful again!
        for k, param in self.params().items():
            l2_loss, l2_grad = l2_regularization(param.value, self.reg)
            loss += l2_loss
            param.grad += l2_grad

        return loss

    def predict(self, X):
        """
        Produces classifier predictions on the set

        Arguments:
          X, np array (test_samples, num_features)

        Returns:
          y_pred, np.array of int (test_samples)
        """
        # TODO: Implement predict
        # Hint: some of the code of the compute_loss_and_gradients
        # can be reused
        pred = np.zeros(X.shape[0], np.int)
        
        res = self.fc1.forward(X)
        res = self.relu1.forward(res)
        res = self.fc2.forward(res)

        shift_pred = res - np.max(res, axis=-1)[:, np.newaxis]
        denom = np.sum(np.exp(shift_pred), axis=-1)[:, np.newaxis]
        probs = np.exp(shift_pred) / denom
        pred = np.argmax(probs, axis=1)
        
        return pred

    def params(self):
        result = {}

        # TODO Implement aggregating all of the params
        result['fc1.W'] = self.fc1.W
        result['fc1.B'] = self.fc1.B
        result['fc2.W'] = self.fc2.W
        result['fc2.B'] = self.fc2.B

        return result
