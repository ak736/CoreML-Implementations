import numpy as np
import pandas as pd

############################################################################
# DO NOT MODIFY CODES ABOVE 
# DO NOT CHANGE THE INPUT AND OUTPUT FORMAT
############################################################################

###### Part 1.1 ######
def mean_square_error(w, X, y):
    """
    Compute the mean square error of a model parameter w on a test set X and y.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing test features
    - y: A numpy array of shape (num_samples, ) containing test labels
    - w: a numpy array of shape (D, )
    Returns:
    - err: the mean square error
    """
    #####################################################
    # TODO 1: Fill in your code here                    #
    #####################################################
    
    # Calculate predictions using the linear model: y_pred = X * w
    y_pred = X.dot(w)
    
    # Calculate the squared differences between predictions and actual values
    squared_diff = (y_pred - y) ** 2

    # Compute the mean of squared differences
    err = np.mean(squared_diff)

    return err

###### Part 1.2 ######
def linear_regression_noreg(X, y):
  """
  Compute the weight parameter given X and y.
  Inputs:
  - X: A numpy array of shape (num_samples, D) containing features
  - y: A numpy array of shape (num_samples, ) containing labels
  Returns:
  - w: a numpy array of shape (D, )
  """
  #####################################################
  #	TODO 2: Fill in your code here                    #
  #####################################################		
  
  
  # Compute w = (X^T * X)^(-1) * X^T * y
  # This is the analytical solution to linear regression
    
  # First calculate X^T * X
  XTX = np.dot(X.T, X)
    
  # Calculate the inverse of X^T * X
  XTX_inv = np.linalg.inv(XTX)
    
  # Calculate X^T * y
  XTy = np.dot(X.T, y)
    
  # Calculate w = (X^T * X)^(-1) * X^T * y
  w = np.dot(XTX_inv, XTy)
  
  return w


###### Part 1.3 ######
def regularized_linear_regression(X, y, lambd):
    """
    Compute the weight parameter given X, y and lambda.
    Inputs:
    - X: A numpy array of shape (num_samples, D) containing features
    - y: A numpy array of shape (num_samples, ) containing labels
    - lambd: a float number specifying the regularization parameter
    Returns:
    - w: a numpy array of shape (D, )
    """
  #####################################################
  # TODO 4: Fill in your code here                    #
  #####################################################		
    
    # Get dimensions
    n, d = X.shape
    
    # Compute w = (X^T * X + lambda * I)^(-1) * X^T * y
    # This is the analytical solution to regularized linear regression
    
    # First calculate X^T * X
    XTX = np.dot(X.T, X)
    
    # Create identity matrix
    I = np.eye(d)
    
    # Calculate X^T * X + lambda * I
    regularized_matrix = XTX + lambd * I
    
    # Calculate the inverse of the regularized matrix
    reg_matrix_inv = np.linalg.inv(regularized_matrix)
    
    # Calculate X^T * y
    XTy = np.dot(X.T, y)
    
    # Calculate w = (X^T * X + lambda * I)^(-1) * X^T * y
    w = np.dot(reg_matrix_inv, XTy)
    return w

###### Part 1.4 ######
def tune_lambda(Xtrain, ytrain, Xval, yval):
    """
    Find the best lambda value.
    Inputs:
    - Xtrain: A numpy array of shape (num_training_samples, D) containing training features
    - ytrain: A numpy array of shape (num_training_samples, ) containing training labels
    - Xval: A numpy array of shape (num_val_samples, D) containing validation features
    - yval: A numpy array of shape (num_val_samples, ) containing validation labels
    Returns:
    - bestlambda: the best lambda you find among 2^{-14}, 2^{-13}, ..., 2^{-1}, 1.
    """
    #####################################################
    # TODO 5: Fill in your code here                    #
    #####################################################		
    # Initialize variables to keep track of best lambda and lowest MSE
    best_lambda = None

    lowest_mse = float('inf')
    
    # Create a list of lambda values to try: 2^-14, 2^-13, ..., 2^-1, 2^0=1
    lambda_values = [2**i for i in range(-14, 1)]
    
    # Try each lambda value
    for lambd in lambda_values:
        # Train model using current lambda
        w = regularized_linear_regression(Xtrain, ytrain, lambd)
        
        # Evaluate model on validation set
        val_mse = mean_square_error(w, Xval, yval)
        
        # If current MSE is better than previous best, update best lambda
        if val_mse < lowest_mse:
            lowest_mse = val_mse
            best_lambda = lambd

    return best_lambda
    

###### Part 1.6 ######
def mapping_data(X, p):
    """
    Augment the data to [X, X^2, ..., X^p]
    Inputs:
    - X: A numpy array of shape (num_training_samples, D) containing training features
    - p: An integer that indicates the degree of the polynomial regression
    Returns:
    - X: The augmented dataset. You might find np.insert useful.
    """
    #####################################################
    # TODO 6: Fill in your code here                    #
    #####################################################		
    
    # Create a copy of X to start with
    X_augmented = X.copy()
    
    # For each power from 2 to p, compute X^power and add to X_augmented
    for power in range(2, p + 1):
        # Compute X^power (element-wise)
        X_power = X ** power
        
        # Concatenate horizontally with existing augmented matrix
        X_augmented = np.hstack((X_augmented, X_power))
    
    return X_augmented

"""
NO MODIFICATIONS below this line.
You should only write your code in the above functions.
"""

