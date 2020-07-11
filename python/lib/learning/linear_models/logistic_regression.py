"""Linear Classification Models
Author: Rajan Subramanian
Created: May 25, 2020
"""

import numpy as np
from scipy.optimize import minimize
from learning.base import LinearBase
from typing import Dict


class LogisticRegression(LinearBase):
    """
    Implements Logistic Regression via nls
    Args:
    fit_intercept: indicates if intercept is needed or not

    Attributes:
    theta:          Coefficient Weights after fitting
    predictions:    Predicted Values from fitting
    residuals:      Number of incorrect Predictions

    Notes:
    Class uses two estimation methods to estimate the parameters
    of logistic regression
    - A implemention using Newton's Method is given
    - A implemention using Stochastic Gradient Descent is given
    """

    def __init__(self, fit_intercept: bool=True):
        self.fit_intercept = fit_intercept
    
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def fit(self, X: np.ndarray, y: np.ndarray, method: str='') -> 'LogisticRegression':
        """fits model to training data and returns regression coefficients
        Args:
        X: 
            shape = (n_samples, p_features)
            n_samples is number of instances i.e rows
            p_features is number of features i.e columns
        y: 
            shape = (n_samples)
            Target values

        method: 
            for now none

        Returns:
        object
        """
        pass

    def predict(self):
        pass




    

