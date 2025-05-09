import numpy as np

#######################################################
# DO NOT MODIFY ANY CODE OTHER THAN THOSE TODO BLOCKS #
#######################################################


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data (either 0 or 1)
    - loss: loss type, either perceptron or logistic
        - w0: initial weight vector (a numpy array)
        - b0: initial bias term (a scalar)
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the final trained weight vector
    - b: scalar, the final trained bias term

    Find the optimal parameters w and b for inputs X and y.
    Use the *average* of the gradients for all training examples
    multiplied by the step_size to update parameters.	
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2

    w = np.zeros(D)
    if w0 is not None:
        w = w0

    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ################################################
        # TODO 1 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize perceptron loss (use -1 as the   #
        # derivative of the perceptron loss at 0)      #
        ################################################

        # Set the random seed to match expected values
        np.random.seed(42)

        # Convert 0/1 labels to -1/+1 for easier handling
        y_signed = 2 * y - 1

        for i in range(max_iterations):
            # Calculate scores for all examples
            scores = np.dot(X, w) + b

            # Initialize gradient accumulators
            dw = np.zeros(D)
            db = 0

            # Calculate gradients for misclassified examples
            for j in range(N):
                # Calculate margin: positive for correct classification, negative for incorrect
                margin = y_signed[j] * scores[j]

                # Update gradients for misclassified examples
                if margin <= 0:  # Misclassified or on boundary
                    dw -= y_signed[j] * X[j]  # Gradient of perceptron loss
                    # Gradient of perceptron loss w.r.t. bias
                    db -= y_signed[j]

            # Update parameters using the average gradient
            w = w - step_size * dw / N
            b = b - step_size * db / N

    elif loss == "logistic":
        ################################################
        # TODO 2 : perform "max_iterations" steps of   #
        # gradient descent with step size "step_size"  #
        # to minimize logistic loss                    #
        ################################################

        for i in range(max_iterations):
            # Calculate predictions using sigmoid
            scores = np.dot(X, w) + b
            predictions = sigmoid(scores)

            # Calculate gradients
            error = predictions - y  # difference between predictions and actual labels

            # Calculate average gradients
            dw = np.dot(X.T, error) / N
            db = np.sum(error) / N

            # Update parameters
            w -= step_size * dw
            b -= step_size * db

    else:
        raise "Undefined loss function."

    assert w.shape == (D,)
    return w, b


def sigmoid(z):
    """
    Inputs:
    - z: a numpy array or a float number

    Returns:
    - value: a numpy array or a float number after applying the sigmoid function 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : fill in the sigmoid function    #
    ############################################

    # Handle potential numerical instability for large negative values
    z_safe = np.clip(z, -500, 500)  # Prevent overflow
    value = 1.0 / (1.0 + np.exp(-z_safe))

    return value


def binary_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model

    Returns:
    - preds: N-dimensional vector of binary predictions (either 0 or 1)
    """
    N, D = X.shape

    #############################################################
    # TODO 4 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################

    # Calculate scores
    scores = np.dot(X, w) + b

    # Threshold at 0
    preds = np.where(scores > 0, 1, 0)

    assert preds.shape == (N,)
    return preds


def multiclass_train(X, y, C,
                     w0=None,
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5,
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data (0, 1, ..., C-1)
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform (stochastic) gradient descent

    Returns:
    - w: C-by-D weight matrix, where C is the number of classes and D 
    is the dimensionality of features.
    - b: a bias vector of length C, where C is the number of classes

    Implement multinomial logistic regression for multiclass 
    classification. Again for GD use the *average* of the gradients for all training 
    examples multiplied by the step_size to update parameters.

    You may find it useful to use a special (one-hot) representation of the labels, 
    where each label y_i is represented as a row of zeros with a single 1 in
    the column that corresponds to the class y_i. Also recall the tip on the 
    implementation of the softmax function to avoid numerical issues.
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0

    b = np.zeros(C)
    if b0 is not None:
        b = b0

    # DO NOT CHANGE THE RANDOM SEED IN YOUR FINAL SUBMISSION
    np.random.seed(42)
    if gd_type == "sgd":

        for it in range(max_iterations):
            n = np.random.choice(N)
            ####################################################
            # TODO 5 : perform "max_iterations" steps of       #
            # stochastic gradient descent with step size       #
            # "step_size" to minimize logistic loss. We already#
            # pick the index of the random sample for you (n)  #
            ####################################################

            # Get the selected sample
            x_n = X[n:n+1]  # Keep 2D shape for matrix operations

            # Calculate scores
            scores = np.dot(x_n, w.T) + b  # Shape (1, C)

            # Compute softmax probabilities
            # Subtract max score for numerical stability
            scores = scores - np.max(scores)
            exp_scores = np.exp(scores)
            probs = exp_scores / np.sum(exp_scores)  # Shape (1, C)

            # Create one-hot encoding of true label
            y_one_hot = np.zeros((1, C))
            y_one_hot[0, y[n]] = 1

            # Compute gradient for this example
            dw = np.dot((probs - y_one_hot).T, x_n)  # Shape (C, D)
            db = (probs - y_one_hot).flatten()  # Shape (C,)

            # Update parameters
            w -= step_size * dw
            b -= step_size * db

    elif gd_type == "gd":
        ####################################################
        # TODO 6 : perform "max_iterations" steps of       #
        # gradient descent with step size "step_size"      #
        # to minimize logistic loss.                       #
        ####################################################

        # Create one-hot encoding for all labels
        y_one_hot = np.zeros((N, C))
        y_one_hot[np.arange(N), y] = 1

        for it in range(max_iterations):
            # Calculate scores for all examples
            scores = np.dot(X, w.T) + b  # Shape (N, C)

            # Apply softmax with numerical stability
            # Subtract max score for numerical stability
            scores = scores - np.max(scores, axis=1, keepdims=True)
            exp_scores = np.exp(scores)
            probs = exp_scores / \
                np.sum(exp_scores, axis=1, keepdims=True)  # Shape (N, C)

            # Compute gradients (average over all examples)
            dw = np.dot((probs - y_one_hot).T, X) / N  # Shape (C, D)
            db = np.sum(probs - y_one_hot, axis=0) / N  # Shape (C,)

            # Update parameters
            w -= step_size * dw
            b -= step_size * db

    else:
        raise "Undefined algorithm."

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained model, C-by-D 
    - b: bias terms of the trained model, length of C

    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Predictions should be from {0, 1, ..., C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    #############################################################
    # TODO 7 : predict DETERMINISTICALLY (i.e. do not randomize)#
    #############################################################

    # Calculate scores for all classes
    scores = np.dot(X, w.T) + b  # Shape (N, C)

    # Predict the class with the highest score
    preds = np.argmax(scores, axis=1)

    assert preds.shape == (N,)
    return preds
