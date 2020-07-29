"""Linear Classification Models
Author: Rajan Subramanian
Created: May 25, 2020
"""

import numpy as np
from scipy.optimize import minimize
from learning.base import LinearBase
from typing import Union, Dict

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
    def __init__(self, fit_intercept: bool=True, degree: int=1):
        self.fit_intercept = fit_intercept
        self.degree = degree 
        self.run = False
    
    def sigmoid(self, z):
        return 1.0 / (1 + np.exp(-z))
    
    def sigmoid_prime(self, z):
        return self.sigmoid(z) * (1 - self.sigmoid(z))
    
    def _jacobian(self, guess: np.ndarray, X: np.ndarray, y: np.ndarray):
        return X @ (y - self.sigmoid(guess))
    
    def _hessian(self, guess: np.ndarray, X: np.ndarray, y: np.ndarray):
        return X @ np.diag()
    
    def _loglikelihood(self, y, z):
        return y @ z - np.log(1 + np.exp(z)).sum()
    
    def _objective_func(self, guess: np.ndarray, X: np.ndarray, y: np.ndarray):
        """the objective function to be minimized, returns weights
        Args:
        guess: 
            initial guess for optimization
            shape = {1, p_features}
            p_features is the numnber of columns of design matrix X
        
        X:
            the design matrix
            shape = {n_samples, n_features}
        
        Returns: 
        Scalar value from loglikelihood function
        """
        # z = X @ theta
        z = self.predict(X, guess=guess)
        f = self._loglikelihood(y, z)
        return f

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

    def predict(self, X: np.ndarray, 
    thetas: Union[np.ndarray, None] = None) -> Union[np.ndarray, Dict]:
        """makes predictions of response variable given input params
        Args: 
        X: 
            shape = (n_samples, p_features)
            n_samples is number of instances
            p_features is number of features
        thetas: 
            if initialized to None: 
                uses estimated theta from fitting process
            if array is given: 
                it serves as initial guess for optimization
    
        Returns: 
        predicted values:
            shape = (n_samples, 1)
        """
        if thetas is None:
            if isinstance(self.theta, np.ndarray):
                return X @ self.theta
            else:
                return X @ self.theta['x']
            return X @ thetas
        




    

