from utils import softmax_cross_entropy, add_momentum, data_loader_mnist, predict_label, DataSplit
import sys
import os
import argparse
import numpy as np
import json

###################################
#   Only modify the TODO blocks   #
###################################


# 1. One linear Neural Network layer with forward and backward steps
class linear_layer:

    """
        The linear (affine/fully-connected) module.

        It is built up with two arguments:
        - input_D: the dimensionality of the input example/instance of the forward pass
        - output_D: the dimensionality of the output example/instance of the forward pass

        It has two learnable parameters:
        - self.params['W']: the W matrix (numpy array) of shape input_D-by-output_D
        - self.params['b']: the b vector (numpy array) of shape 1-by-output_D

        It will record the partial derivatives of loss w.r.t. self.params['W'] and self.params['b'] in:
        - self.gradient['W']: input_D-by-output_D numpy array
        - self.gradient['b']: 1-by-output_D numpy array
    """

    def __init__(self, input_D, output_D):

        self.params = dict()
        self.gradient = dict()

        ###############################################################################################
        # TODO: Use np.random.normal() with mean 0 and standard deviation 0.1 to initialize
        #   - self.params['W']
        #   - self.params['b']
        ###############################################################################################

        # Initialize weights with random normal distribution (mean=0, std=0.1)
        self.params['W'] = np.random.normal(0, 0.1, (input_D, output_D))
        # Initialize bias with random normal distribution (mean=0, std=0.1)
        self.params['b'] = np.random.normal(0, 0.1, (1, output_D))

        ###############################################################################################
        # TODO: Initialize the following two (gradients) with zeros
        #   - self.gradient['W']
        #   - self.gradient['b']
        ###############################################################################################

        # Initialize gradients with zeros
        self.gradient['W'] = np.zeros((input_D, output_D))
        self.gradient['b'] = np.zeros((1, output_D))

    def forward(self, X):
        """
            The forward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, where each 'row' is an input example/instance (N is the batch size)

            Return:
            - forward_output: A N-by-output_D numpy array, where each 'row' is an output example/instance.
        """

        ################################################################################
        # TODO: Implement the linear forward pass. Store the result in forward_output  #
        ################################################################################

        # Linear transformation: forward_output = X * W + b
        forward_output = np.dot(X, self.params['W']) + self.params['b']

        return forward_output

    def backward(self, X, grad):
        """
            The backward pass of the linear (affine/fully-connected) module.

            Input:
            - X: A N-by-input_D numpy array, the input to the forward pass.
            - grad: A N-by-output_D numpy array, where each 'row' (say row i) is the partial derivative of the mini-batch loss
                 w.r.t. forward_output[i].

            Operation:
            - Compute the partial derivatives (gradients) of the mini-batch loss w.r.t. self.params['W'], self.params['b'].

            Return:
            - backward_output: A N-by-input_D numpy array, where each 'row' (say row i) is the partial derivatives of the mini-batch loss w.r.t. X[i].
        """

        #################################################################################################
        # TODO: Implement the backward pass (i.e., compute the following three terms)
        #   - self.gradient['W'] (input_D-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['W'])
        #   - self.gradient['b'] (1-by-output_D numpy array, the gradient of the mini-batch loss w.r.t. self.params['b'])
        #   - backward_output (N-by-input_D numpy array, the gradient of the mini-batch loss w.r.t. X)
        # only return backward_output, but need to compute self.gradient['W'] and self.gradient['b']
        #################################################################################################

        # Gradient with respect to W: dL/dW = X^T * dL/dY
        self.gradient['W'] = np.dot(X.T, grad)

        # Gradient with respect to b: dL/db = sum(dL/dY, axis=0)
        self.gradient['b'] = np.sum(grad, axis=0, keepdims=True)

        # Gradient with respect to X: dL/dX = dL/dY * W^T
        backward_output = np.dot(grad, self.params['W'].T)

        return backward_output


# 2. ReLU Activation
class relu:

    """
        The relu (rectified linear unit) module.

        It is built up with NO arguments.
        It has no parameters to learn.
        self.mask is an attribute of relu. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self):
        self.mask = None

    def forward(self, X):
        """
            The forward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape.

            Return:
            - forward_output: A numpy array of the same shape of X
        """

        ################################################################################
        # TODO: Implement the relu forward pass. Store the result in forward_output    #
        ################################################################################

        # ReLU activation: f(x) = max(0, x)
        # Store which elements are positive for use in backward pass
        self.mask = (X > 0)
        forward_output = np.maximum(0, X)

        return forward_output

    def backward(self, X, grad):
        """
            The backward pass of the relu (rectified linear unit) module.

            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss
                 w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """

        ####################################################################################################
        # TODO: Implement the backward pass
        # You can use the mask created in the forward step.
        ####################################################################################################

        # ReLU gradient: f'(x) = 1 if x > 0, 0 otherwise
        backward_output = grad * self.mask

        return backward_output


# 3. tanh Activation
class tanh:

    def forward(self, X):
        """
            Input:
            - X: A numpy array of arbitrary shape.

            Return:
            - forward_output: A numpy array of the same shape of X
        """

        ################################################################################
        # TODO: Implement the tanh forward pass. Store the result in forward_output
        # You can use np.tanh()
        ################################################################################

        # Tanh activation: f(x) = tanh(x)
        forward_output = np.tanh(X)

        return forward_output

    def backward(self, X, grad):
        """
            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in forward_output.

            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in  X.
        """

        ####################################################################################################
        # TODO: Implement the backward pass
        # Derivative of tanh(z) is (1 - tanh(z)^2)
        ####################################################################################################

        # Tanh gradient: f'(x) = 1 - tanh^2(x)
        tanh_output = np.tanh(X)
        backward_output = grad * (1 - tanh_output**2)

        return backward_output


# 4. Dropout
class dropout:

    """
        It is built up with one argument:
        - r: the dropout rate

        It has no parameters to learn.
        self.mask is an attribute of dropout. You can use it to store things (computed in the forward pass) for the use in the backward pass.
    """

    def __init__(self, r):
        self.r = r
        self.mask = None

    def forward(self, X, is_train):
        """
            Input:
            - X: A numpy array of arbitrary shape.
            - is_train: A boolean value. If False, no dropout should be performed.

            Operation:
            - Suppose p is uniformly randomly generated from [0,1]. If p >= self.r, output that element multiplied by (1.0 / (1 - self.r)); otherwise, output 0 for that element

            Return:
            - forward_output: A numpy array of the same shape of X (the output of dropout)
        """

        ################################################################################
        #  TODO: We provide the forward pass to you. You only need to understand it.   #
        ################################################################################

        if is_train:
            self.mask = (np.random.uniform(0.0, 1.0, X.shape) >=
                         self.r).astype(float) * (1.0 / (1.0 - self.r))
        else:
            self.mask = np.ones(X.shape)
        forward_output = np.multiply(X, self.mask)
        return forward_output

    def backward(self, X, grad):
        """
            Input:
            - X: A numpy array of arbitrary shape, the input to the forward pass.
            - grad: A numpy array of the same shape of X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in forward_output.


            Return:
            - backward_output: A numpy array of the same shape as X, where each element is the partial derivative of the mini-batch loss w.r.t. the corresponding element in X.
        """

        ####################################################################################################
        # TODO: Implement the backward pass
        # You can use the mask created in the forward step
        ####################################################################################################

        # We use the same mask that was created in the forward pass
        # The mask already contains the scaling factor 1/(1-r) for non-dropped elements
        backward_output = np.multiply(grad, self.mask)

        return backward_output


# 5. Mini-batch Gradient Descent Optimization
def miniBatchGradientDescent(model, momentum, _alpha, _learning_rate):

    for module_name, module in model.items():

        # check if a module has learnable parameters
        if hasattr(module, 'params'):
            for key, _ in module.params.items():
                # This is the gradient for the parameter named "key" in this module
                g = module.gradient[key]

                if _alpha <= 0.0:
                    ####################################################################################
                    # TODO: update the model parameter module.params[key] by a step of gradient descent.
                    # Note again that the gradient is stored in g already.
                    ####################################################################################
                    # Standard gradient descent update: param = param - learning_rate * gradient
                    module.params[key] = module.params[key] - \
                        _learning_rate * g

                else:
                    ###################################################################################################
                    # TODO: Update the model parameter module.params[key] by a step of gradient descent with momentum.
                    # Access the previous momentum by momentum[module_name + '_' + key], and then update it directly.
                    ###################################################################################################
                    # Get the momentum key for this parameter
                    momentum_key = module_name + '_' + key

                    # Update momentum: v_t = alpha * v_{t-1} - learning_rate * gradient
                    momentum[momentum_key] = _alpha * \
                        momentum[momentum_key] - _learning_rate * g

                    # Update parameter: param = param + momentum
                    module.params[key] = module.params[key] + \
                        momentum[momentum_key]

    return model


def main(main_params):

    ### set the random seed. DO NOT MODIFY. ###
    np.random.seed(int(main_params['random_seed']))

    ### data processing ###
    Xtrain, Ytrain, Xval, Yval, _, _ = data_loader_mnist(
        dataset=main_params['input_file'])
    N_train, d = Xtrain.shape
    N_val, _ = Xval.shape

    index = np.arange(10)
    unique, counts = np.unique(Ytrain, return_counts=True)
    counts = dict(zip(unique, counts)).values()

    trainSet = DataSplit(Xtrain, Ytrain)
    valSet = DataSplit(Xval, Yval)

    ### building/defining MLP ###
    """
    In this script, we are going to build a MLP for a 10-class classification problem on MNIST.
    The network structure is input --> linear --> relu --> dropout --> linear --> softmax_cross_entropy loss
    the hidden_layer size (num_L1) is 1000
    the output_layer size (num_L2) is 10
    """
    model = dict()
    num_L1 = 1000
    num_L2 = 10

    # experimental setup
    num_epoch = int(main_params['num_epoch'])
    minibatch_size = int(main_params['minibatch_size'])

    # optimization setting
    _learning_rate = float(main_params['learning_rate'])
    _step = 10
    _alpha = float(main_params['alpha'])
    _dropout_rate = float(main_params['dropout_rate'])
    _activation = main_params['activation']

    if _activation == 'relu':
        act = relu
    else:
        act = tanh

    # create objects (modules) from the module classes
    model['L1'] = linear_layer(input_D=d, output_D=num_L1)
    model['nonlinear1'] = act()
    model['drop1'] = dropout(r=_dropout_rate)
    model['L2'] = linear_layer(input_D=num_L1, output_D=num_L2)
    model['loss'] = softmax_cross_entropy()

    # Momentum
    if _alpha > 0.0:
        momentum = add_momentum(model)
    else:
        momentum = None

    train_acc_record = []
    val_acc_record = []

    train_loss_record = []
    val_loss_record = []

    ### run training and validation ###
    for t in range(num_epoch):
        print('At epoch ' + str(t + 1))
        if (t % _step == 0) and (t != 0):
            _learning_rate = _learning_rate * 0.1

        idx_order = np.random.permutation(N_train)

        train_acc = 0.0
        train_loss = 0.0
        train_count = 0

        val_acc = 0.0
        val_count = 0
        val_loss = 0.0

        for i in range(int(np.floor(N_train / minibatch_size))):

            # get a mini-batch of data
            x, y = trainSet.get_example(
                idx_order[i * minibatch_size: (i + 1) * minibatch_size])

            ### forward pass ###
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=True)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)

            ### backward pass ###
            grad_a2 = model['loss'].backward(a2, y)
            ######################################################################################
            # TODO: Call the backward methods of every layer in the model in reverse order.
            # We have given the first and last backward calls (above and below this TODO block).
            ######################################################################################
            # Complete the backward pass chain in reverse order of the forward pass
            # Forward pass order was: L1 -> nonlinear1 -> drop1 -> L2 -> loss
            # So backward pass order should be: loss -> L2 -> drop1 -> nonlinear1 -> L1

            # L2 layer backward pass: input was d1, output was a2, incoming gradient is grad_a2
            grad_d1 = model['L2'].backward(d1, grad_a2)

            # Dropout layer backward pass: input was h1, output was d1, incoming gradient is grad_d1
            grad_h1 = model['drop1'].backward(h1, grad_d1)

            # Nonlinear (ReLU/tanh) layer backward pass: input was a1, output was h1, incoming gradient is grad_h1
            grad_a1 = model['nonlinear1'].backward(a1, grad_h1)

            # The L1 backward pass is already provided:
            grad_x = model['L1'].backward(x, grad_a1)

            ### gradient_update ###
            model = miniBatchGradientDescent(
                model, momentum, _alpha, _learning_rate)

        ### Computing training accuracy and obj ###
        for i in range(int(np.floor(N_train / minibatch_size))):

            x, y = trainSet.get_example(
                np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            ### forward pass ###
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=False)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)
            train_loss += loss
            train_acc += np.sum(predict_label(a2) == y)
            train_count += len(y)

        train_acc = train_acc / train_count
        train_acc_record.append(train_acc)
        train_loss_record.append(train_loss)

        print('Training loss at epoch ' + str(t + 1) + ' is ' + str(train_loss))
        print('Training accuracy at epoch ' +
              str(t + 1) + ' is ' + str(train_acc))

        ### Computing validation accuracy ###
        for i in range(int(np.floor(N_val / minibatch_size))):

            x, y = valSet.get_example(
                np.arange(i * minibatch_size, (i + 1) * minibatch_size))

            ### forward pass ###
            a1 = model['L1'].forward(x)
            h1 = model['nonlinear1'].forward(a1)
            d1 = model['drop1'].forward(h1, is_train=False)
            a2 = model['L2'].forward(d1)
            loss = model['loss'].forward(a2, y)
            val_loss += loss
            val_acc += np.sum(predict_label(a2) == y)
            val_count += len(y)

        val_loss_record.append(val_loss)
        val_acc = val_acc / val_count
        val_acc_record.append(val_acc)

        print('Validation accuracy at epoch ' +
              str(t + 1) + ' is ' + str(val_acc))

    # save file
    json.dump({'train': train_acc_record, 'val': val_acc_record},
              open('MLP_lr' + str(main_params['learning_rate']) +
                   '_m' + str(main_params['alpha']) +
                   '_d' + str(main_params['dropout_rate']) +
                   '_a' + str(main_params['activation']) +
                   '.json', 'w'))

    print('Finish running!')
    return train_loss_record, val_loss_record


if __name__ == "__main__":

    ######################################################################################
    # These are the default arguments used to run your code.
    # These parameters will be changed while grading.
    # You can modify them to test your code (this does not affect the grading as long as
    # you remember to run runme.py before submitting).
    ######################################################################################

    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default=42)
    parser.add_argument('--learning_rate', default=0.01)
    parser.add_argument('--alpha', default=0.0)
    parser.add_argument('--dropout_rate', default=0.5)
    parser.add_argument('--num_epoch', default=10)
    parser.add_argument('--minibatch_size', default=5)
    parser.add_argument('--activation', default='relu')
    parser.add_argument('--input_file', default='mnist_subset.json')
    args = parser.parse_args()
    main_params = vars(args)
    main(main_params)
