"""Generalized Linear Models
Author: Rajan Subramanian
Created: May 23, 2020
"""

import numpy as np 
from abc import ABCMeta, abstractmethod


class Base(metaclass=ABCMeta):
    """Abstract Base class representing the Linear Model"""
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass


class LinearRegression(Base):
    """
    Implements the classic Linear Regression via ols
    Args:
       None

       Attributes:
       theta:           Coefficient Weights after fitting
       errors:          Number of Incorrect Predictions
    """

    def __init__(self, fit_intercept: bool=True):
        self.fit_intercept = fit_intercept

    def make_constant(self, X: np.ndarray) -> np.ndarray: 
        if self.fit_intercept: 
            ones = np.ones(shape=(X.shape[0], 1))
            return np.concatenate((ones, X), axis=1)
        else:
            return X

    def fit(X: np.ndarray, y: np.ndarray, method: str='ols') -> 'LinearRegression':
        """fits training data via ordinary least Squares (ols)
            given by: theta = (X'X)^-1 X'y

        Args:
        X: shape = (n_samples, p_features)
                    n_samples is number of instances i.e rows
                    p_features is number of features
        y: shape = (n_samples)
                    Target values

        method: 
            the fitting procedure, default to computing matrix inverse

        Returns:
        object
        """
        n = X.shape[0]
        X = self.make_constant(X)
        self.theta = np.dot(np.dot(np.linalg.inv(X.T.dot(X)), X.T), y)
        self.errors = ((y - self.predict(X))**2)
        self.rss = self.errors.sum() 
        self.standard_error = np.sqrt(self.rss / (n - 2))
        self.covar = (self.standard_error ** 2) * np.inv(X.T.dot(X))
        return self

    def predict(self, X: np.ndarray):
        return np.dot(self.theta, X)

