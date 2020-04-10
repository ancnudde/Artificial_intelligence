import numpy as np
from mnist_prepare import parse_data


def sigmoid_activation(neuron_layer):
    """
    Activates the neurons layer with a sigmoid or logistics function.

    Uses the sigmoid function 1 / (1 + e^z) to activate the neuron layer.

    Args:
       neuron_layer: Layer of neurons to be activated.

    Returns:
        The activated neurons layer.

    """
    return (1.0 / (1.0 + np.exp(-neuron_layer)))


def sigmoid_derivative(neuron_layer):
    """
    Computes the derivative of the sigmoid activation function for a given
    layer.

    The derivative of the sigmoid function is given by:

        sigmoid(z) * (1 - sigmoid(z))

    Args:
       neuron_layer: Layer of neurons to be activated.

    Returns:
        The neuron layer activated by the derivative of the sigmoid function.

    """
    activated_inputs = sigmoid_activation(neuron_layer)
    return activated_inputs * (1 - activated_inputs)


class ShallowNetwork:
    """
    Creates a feedforward neural network model with one hidden layer.
    """

    def __init__(self, architecture, activation_func=sigmoid_activation):
        """
        Initializes the perceptron.

        The perceptron is here a Single Layer Perceptron with one hidden
        layer.
        The biases are generated randomly from a gaussian distribution with
        standard deviation 1, and the weights are generated from this same
        distribution but are divided by the number of inputs.This trick
        sharpens the distribution around 0 and makes the neurons less likely
        to saturate, what slows the learning process.

        Args:
           architecture: Architecture of the network, as a tuple. The
                         first value of the tuple gives the number of input
                         nodes, the second the number of hidden neurons, and
                         the tird the number of output neurons.
           activation_function: Activation function to use for neurons
                                activation.

        """
        self.architecture = architecture
        # Generates the biases as drawn from gaussian.
        # distribution with standard deviation = 1.
        self.biases = [np.random.randn(y, 1)
                       for y in self.architecture[1:]]
        # Generates the weights as drawn from gaussian distribution with
        # standard deviation = 1 and divided by the number of inputs.
        # This reduces the risk of the neuron to saturate.
        self.weights = [np.random.randn(y, x)/np.sqrt(x)
                        for x, y in zip(self.architecture[:-1],
                                        self.architecture[1:])]
        self.activation_function = activation_func
        # neurons values, used for backpropagation computation.
        self.z_values = []
        self.activations = []

    def feedforward_computation(self, input_vector):
        """
        Computes the output of the neuron.

        Uses the feedforward algorithm to compute the value of the output
        for the given weights and biases. Computes the weighted sum of the
        neurons of the previous layer and uses the activation function to
        activate it, for each neuron of the current layer.

        Args:
           input_vector: Values of the input as an array or list.

        Returns:
            The activated neurons values of the output layer.
            Updates the 'activations' attribute of the network.

        """
        # Initialisation.
        self.z_values = []
        self.activations = [input_vector]
        next_layer = input_vector
        # Iteration over each layer.
        for bias_layer, weight_layer in zip(self.biases, self.weights):
            # Get the neuron value.
            z = np.dot(weight_layer, next_layer) + bias_layer
            self.z_values.append(z)  # Stocked for backward pass.
            # Activates the neuron.
            next_layer = self.activation_function(z)
            self.activations.append(next_layer)  # Stocked for backward pass.
        # Returns the output layer activations.
        return (next_layer)

    def error_cost(self, output, label):
        """
        Computes the error of output layer.

        This function uses the cross-entropy as a cost function to
        assess the error on the output. Cross entropy is given by:

            C = -Sum(y * ln(a) + (1 - y) * ln(1 - a)).

        This cost function gives better results than the MSE by avoiding
        a learning slowdown that is encountered by using the quadratic cost
        function.

        Args:
           output: Values of the activated output layer.
           label: Expected output of the network given the input.

        Returns:
            The error on the predicted output.

        """
        # nan_to_num avoid 'nan's that can be obtained when the predicted
        # and disered output have a 1 in the same position, and replace it
        # by the value 0.0.
        return (np.sum(np.nan_to_num(-label * np.log(output) -
                                     (1 - label) * np.log(1 - output))))

    def error_rate(self, output, label):
        """
        Computes the error rate, used in the partial derivative for the
        gradient descent.

        Args:
           output: Values of the output layer.
           label: Expected output of the network given the input.

        Returns:
            The error rate of the network.

        """
        return (output - label)

    def backpropagation(self, input_vector, label):
        """
        Applies backpropagation to adapt the weights and biases for the
        network.

        Backpropagation uses the error on the output compared to the expected
        output to compute new biases and weights for the network, allowing the
        model to learn. To do so, we need:

            1) To compute the error on the output node. It is denoted
               nabla_a * C, a vector that represents the rate of change of
               the cost-function given the activations of the output neurons.
               Using cross-entropy cost function as here, the error on the
               output is given by

                   delta^L = (a^L - y),

               with a the activations on the output layer, and y the expected
               output.

            2) Propagate the output error to the previous layers. This is
               done using the general formula for the previous layer l:

                   delta^l = ((w^(l+1)^T * delta^(l+1) . sigma'(z^l);

                with delta^l the error on layer l, delta^(l+1) the error on
                layer l + 1 and the sigma term the derivative of the activation
                function on the neurons of layer l. w^(l+1)^T is the transposed
                of the weights of layer l+1.

            3) Compute the new biases and weights based on these errors. We
               compute them as the derivatives of the cost function with
               respect to the variation in biases and weights respectively

                For the weights:

                    der(C)/der(b) = delta^l

                For the biases:

                    der(C)/der(w) = a^(l-1) * delta^l

        Args:
           input_vector: Values of the input as an array or list.
           label: Expected output of the network given the input.

        Returns:
            Gradients of the cost function with respect to the biases and
            weights for each layer.

        """
        # Forward pass.
        self.feedforward_computation(input_vector)
        # Computation for the output layer;
        # Computes the error rate = delta^L.
        delta_l = self.error_rate(self.activations[-1], label)
        # Initialize the gradients for each layer.
        gradient_biases = [np.zeros(bias.shape)
                           for bias in self.biases]
        gradient_weights = [np.zeros(weight.shape)
                            for weight in self.weights]
        # Gradients for the last layer.
        gradient_weights[-1] = np.dot(delta_l, self.activations[-2].T)
        gradient_biases[-1] = delta_l
        # Computation for previous layers.
        for layer in range(2, len(self.architecture)):
            # delta^l computation
            delta_l = np.dot(
                self.weights[-layer + 1].T, delta_l) *\
                sigmoid_derivative(self.z_values[-layer])
            # Gradients computation.
            gradient_biases[-layer] = delta_l
            gradient_weights[-layer] =\
                np.dot(delta_l, self.activations[-layer - 1].T)
        return (gradient_biases, gradient_weights)

    def parameters_update(self, batch, eta, lbda, n):
        """
        Executes the weights and biases updates for the backpropagation.

        Computes the errors for each inpu using the chain rule. The weights and
        biases of the network are updated in the body of the function. This
        step incorporates the L2 regularization, in which the updates are
        scaled by a factor (1 - eta) * lambda / n, with eta the learning rate,
        lambda the regularization parameter telling how much the high weights
        should be scaled, and n the size of the full dataset.

        Args:
           batch: Mini-batch used to update the weights and biases.
           eta: Learning rate, defines 'how fast' the network learns from a
                single batch.
           lbda: Regularization parameter, defines how much the large weights
                 should be reduced to avoid overfitting.
           n: Size of the full dataset.

        """
        # Initializes the weights and biases gradients containers.
        gradient_biases = [np.zeros(bias.shape) for bias in self.biases]
        gradient_weights = [np.zeros(weight.shape) for weight in self.weights]
        # For each input in the mini-batch;
        for inp, outp in batch:
            # Backpropagates the error.
            update_grd_biases, update_grd_weights = self.backpropagation(inp,
                                                                         outp)
            # Keeps the gradients to weights and biases.
            gradient_biases = [gd_b + up_b for gd_b, up_b in zip(
                gradient_biases, update_grd_biases)]
            gradient_weights = [gd_w + up_w for gd_w, up_w in zip(
                gradient_weights, update_grd_weights)]
        # Updates the biases and gradients for the mini-batch.
        self.weights = [(1-eta*(lbda/n))*weight - (eta / len(batch))
                        * gradient_w
                        for weight, gradient_w in zip(self.weights,
                                                      gradient_weights)]
        self.biases = [bias - (eta / len(batch)) * gradient_b
                       for bias, gradient_b in zip(self.biases,
                                                   gradient_biases)]

    def evaluation(self, test_data):
        """
        Evaluate the accuracy of classification.

        /!\ Adapted for the mnist classification problem (handrwitten digit
        recognition) with output on the form of a vector filled with a one in
        the position corresponding to the predicted value, and zeros everywhere
        else.
        Should be adapted to the expected output for specific problem. /!\

        Args:
           test_data: Tuple containing in first position the flattened image
           as a column matrix, and in second position the expected output
           encoded as a column matrix with a 1 in position corresponding to
           the expected value, and 0 everywhere else.

        Returns:
            The number of correctly predicted elements.

        """
        # Gets the predicted and expected values.
        test_results = [(np.argmax(self.feedforward_computation(x)),
                         np.argmax(y))
                        for (x, y) in test_data]
        # Computes the number of correctly predicted values.
        return sum(int(x == y) for (x, y) in test_results)

    def stochastic_gradient_descent(self, dataset, n_epoch,
                                    batch_size, eta, lbda,
                                    test_data=None):
        """
        Trains the network using a stochastic gradient descent.

        In stochastic gradient descent, we train the network by dividing the
        dataset into mini-batches. Each minibatch contains a certain number of
        inputs, and each mini-batch is used to update the weights and biases.
        Using mini-batches allows to improve the learning compared to full-
        batch, and is more computationaly efficient than using inputs one by
        one.

        This process applies L2 or Ridge regularisation. This improvement
        scales the weights and reduce the importance of larges weights,
        allowing a better generalization (reduction of the overfitting).
        This regularisation is achieved by scaling the weights in the weights
        update step by a new term:

            1 - eta * (lambda / n)

        with lambda the regularization parameter; the larger it is, the more
        we reduce the weights.

        Args:
           dataset: Set of inputs to use to train the network.
           n_epoch: Number of epochs (number of times to revisit the whole
                    dataset) to train the network.
           batch_size: Size of the mini-batches.
           eta: Learning rate, determines how fast the network 'learns' from a
                batch.
           lbda: Regularization parameter, defines how much the large weights
                 should be reduced to avoid overfitting.
            test_data: Dataset used for testing the network.

        Returns:
            The trained network.

        """
        for epoch in range(n_epoch):
            n = len(dataset)
            # Shuffles the dataset to get random batches.
            np.random.shuffle(dataset)
            # Divides the dataset in mini-batches.
            mini_batches = [dataset[i: i + batch_size]
                            for i in range(0, len(dataset), batch_size)]
            # For each batch mini-batch;
            for batch in mini_batches:
                # Updates the parameters.
                self.parameters_update(batch, eta, lbda, n)
            # Tests the accuracy on the test set.
            if test_data:
                n_test = len(test_data)
                print("Epoch {0}: {1} / {2}".format(
                    epoch, self.evaluation(test_data), n_test))

    def predict(self, target):
        """
        Predicts the value for a single input.

        /!\  Suits the mnist problem encoding, can have to be adapted
             in other situations.
        """
        return(np.argmax(self.feedforward_computation(target)))


data = parse_data('mnist.pkl', d_set='train')
t_data = parse_data('mnist.pkl', d_set='test')
nn = ShallowNetwork([784, 30, 10], )
nn.stochastic_gradient_descent(data, 30, 10, 0.5, 5.0, t_data)
