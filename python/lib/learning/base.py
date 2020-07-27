"""Abstract base class for various models"""

from abc import ABCMeta, abstractmethod
from typing import Dict
from sklearn.datasets import make_regression
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from operator import itemgetter

class LinearBase(metaclass=ABCMeta):
    """Abstract Base class representing the Linear Model"""
    @abstractmethod
    def fit(self):
        pass

    @abstractmethod
    def predict(self):
        pass

    def make_regression_example(self, n_samples: int=1000, n_features: int=5) -> Dict:
        features, output, coef = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=n_features, n_targets=1,
            noise = 5, coef=True)
        return dict(zip(['X','y','coef'], [features, output, coef]))
    
    def make_constant(self, X: np.ndarray) -> np.ndarray:
        raise DeprecationWarning("function deprecated -- use make_polynomial instead")
        if self.fit_intercept: 
            ones = np.ones(shape=(X.shape[0], 1))
            return np.concatenate((ones, X), axis=1)
        return X
    
    def make_polynomial(self, X: np.ndarray) -> np.ndarray: 
        degree, bias = self.degree, self.fit_intercept 
        pf = PolynomialFeatures(degree=degree, include_bias=bias)
        return pf.fit_transform(X)

    def reg_plot(self, X, y):
        plt.figure(figsize=(10,6))
        plt.scatter(X, y)
        # sort by design matrix -- needed for matplotlib
        sorted_values = iter(sorted(zip(X.flatten(), self.predictions), key=itemgetter(0)))
        X, pred = zip(*sorted_values)
        plt.plot(X, pred,'m-')
        plt.title("Regression Plot")