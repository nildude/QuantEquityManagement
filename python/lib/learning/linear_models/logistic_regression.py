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
    def __init__(self, fit_intercept: bool =True, degree: int = 1):
        self.fit_intercept = fit_intercept
        self.degree = degree
        self.run = False

    def sigm(self, z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function

        Args:
            z (np.ndarray): input value from linear transformation

        Returns:
            np.ndarray: sigmoid function value
        """
        return 1.0 / (1 + np.exp(-z))

    def sigm_prime(self, z: np.ndarray) -> np.ndarray:
        """computes the first derivative of sigmoid function

        Args:
            z (np.ndarray): input value from linear transformation

        Returns:
            np.ndarray: the first derivative of sigmoid function
        """
        return self.sigmoid(z) * (1 - self.sigmoid(z))

    def _jacobian(self,
                  guess: np.ndarray,
                  X: np.ndarray,
                  y: np.ndarray
                  ) -> np.ndarray:
        """Computes the jacobian
        Args:
            guess (np.ndarray): the initial guess for optimizer
            X (np.ndarray): design matrix, shape = (n_samples, n_features)
            y (np.ndarray): response variable, shape = (n_samples,)

        Returns:
            [np.ndarray]: first partial derivatives wrt weights
        """
        return X @ (y - self.sigmoid(guess))

    def _hessian(self,
                 guess: np.ndarray,
                 X: np.ndarray,
                 y: np.ndarray
                 ) -> np.ndarray:
        """computes the hessian wrt weights

        Args:
            guess (np.ndarray): initial guess for optimizer
            X (np.ndarray): design matrix, shape = (n_samples, n_features)
            y (np.ndarray): response variable, shape = (n_samples,)

        Returns:
            np.ndarray: Hessian wrt weights
        """
        pass

    def _loglikelihood(self, y, z):
        """returns the loglikelihood function of logistic regression

        Args:
            y (np.ndarray): response variable, shape = (n_samples,)
            z (np.ndarray): result of net input function

        Returns:
            np.ndarray: loglikelihood function
        """
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

    def net_input(self, X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Computes linear transformation X*theta

        Args:
            X : the coefficient matrix
            thetas : the initial weights of logistic function

        Returns:
            the linear transformation
        """
        return X @ thetas

    def predict(self,
                X: np.ndarray,
                thetas: np.ndarray = None,
                ) -> Union[np.ndarray, Dict]:
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
                return self.net_input(X, self.theta)
            else:
                return self.net_input(X, self.theta['x'])
            return self.net_input(X, thetas)