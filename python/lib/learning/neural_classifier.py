"""
Implements the Perception & Adaline Learning Algorithm
Author: Rajan Subramanian
Created: May 18, 2010
"""

import numpy as np
import matplotlib.pyplot as plt

class Perception:
    """first artifical neural classifier
       
       Args:
       eta:             Learning rate (between 0.0 and 1.0)
       n_iter:          passees over the training set
       random_state:    Random Number Generator seed
                        for random weight initilization

       Attributes:
       w:               Weights after fitting
       errors:          Number of misclassifications(updates) in each epoch
    """
    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'Perception':
        """fits training data
        Args:
        X: shape = {n_samples, p_features}
                    n_samples is number of instances i.e rows
                    p_features is number of features (the dimension of dataset)

        y: shape = {n_samples, 1}
                    Target values

        Returns:
        object
        """
        # random number gen seed to reproduce values if needed
        rgen = np.random.RandomState(self.random_state)
        # initialize weights from normal distribution
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.errors = []
        for _ in range(self.n_iter):
            errors = 0
            # for each instance in training set
            for xi, target in zip(X, y):
                # calculate the weight update by perception rule
                delta_wj = self.eta * (target - self.predict(xi))
                # update all weights simultaneously
                # given by wj := wj + delta_wj
                self.w[1:] += + delta_wj * xi
                # since x0 = 1 by construction
                self.w[0] += delta_wj 
                # calculate number of misclassifications
                errors += int(delta_wj != 0)
            self.errors.append(errors)
        return self 

    def net_input(self, X: np.ndarray) -> np.ndarray:
        """computes the net input vector
            z = w1x1 + w2x2 + ... + wpXp
        Args:
        X: shape = {n_samples, p_features}
                    n_samples is # of instances
                    p_features is number of features (dimension of dataset)
        Returns:
        z: shape = {n_samples, 1}
                    net input vector
        """
        return np.dot(X, self.w[1:]) + self.w[0]

    def predict(self, X: np.ndarray) -> float:
        """
        computes the classifier phi(z) 
        where phi(z) = 1 if z:= w'x >=0, -1 otherwise

        Args:
        X: shape {n_samples, p_features}

        Returns:
        classifier with value +1 or -1
        """
        return np.where(self.net_input(X) > 0, 1, -1)

    def plot_misclassifications(self) -> None:
        """plots the misclassifications given the number of epoochs
            requires to call the fit() first
        """
        try:
            plt.plot(range(1, self.n_iter + 1), self.errors, marker='o');
            plt.xlabel("epoch")
            plt.ylabel("# of misclassifications")
        except AttributeError as e: 
            print("must call fit() first before plotting misclassifications")
        else:
            return 

class AdalineGD:
    """artificial neural classifier 
        implemented with gradient descent
       
       Args:
       eta:             Learning rate (between 0.0 and 1.0)
       n_iter:          passees over the training set
       random_state:    Random Number Generator seed
                        for random weight initilization

       Attributes:
       w:               Weights after fitting
       errors:          Number of misclassifications(updates) in each epoch
    """
    def __init__(self, eta: float = 0.01, n_iter: int = 50, random_state: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'AdalineGD':
        """fits training data
        Args:
        X: shape = {n_samples, p_features}
                    n_samples is number of instances i.e rows
                    p_features is number of features (the dimension of dataset)

        y: shape = {n_samples, 1}
                    Target values

        Returns: 
        object
        """
        # random number gen seed to reproduce values if needed
        rgen = np.random.RandomState(self.random_state)
        # initialize weights from normal distribution
        self.w = rgen.normal(loc=0.0, scale=0.01, size=1+X.shape[1])
        self.cost = []
        for _ in range(self.n_iter):
            # calculate net input
            net_input = self.net_input(X)
            # calculate the linear activation function phi(z) = w'x = z
            output = self.activation(net_input)
            errors = y - output 
            # update the weights
            self.w[1:] += eta * X.T.dot(errors)
            self.w[0] += eta * errors.sum()
            # sse based on J(w) = 1/2 sum(yi - yhat)**2
            cost = (errors**2).sum() / 2.0 
            self.cost.append(cost)
        return self 



    def net_input(self, X: np.ndarray) -> np.ndarray:
        """computes the net input vector
            z = w1x1 + w2x2 + ... + wpXp
        Args:
        X: shape = {n_samples, p_features}
                    n_samples is # of instances
                    p_features is number of features (dimension of dataset)
        Returns:
        z: shape = {n_samples, 1}
                    net input vector
        """
        return np.dot(X, self.w[1:]) + self.w[0]

    def activation(self, X: np.ndarray) -> np.ndarray:
        """compute linear activation z = w'x = phi(z)
        Args: 
        X: shape = {n_samples, n_features}
        
        Returns:
        the input by itself
        """
        return X

    def predict(self, X: np.ndarray) -> float:
        """
        computes the classifier phi(z) 
        where phi(z) = 1 if z:= w'x >=0, -1 otherwise

        Args:
        X: shape {n_samples, p_features}

        Returns:
        classifier with value +1 or -1
        """
        return np.where(self.activation(self.net_input(X)) > 0, 1, -1)