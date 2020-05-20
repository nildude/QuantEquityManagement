"""Implements the Perception Learning Algorithm
Author: Rajan Subramanian
Created: May 18, 2010
"""
import numpy as np 

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

    def fit(self, X: np.array(), y: np.array()) -> self:
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
        self.w = rgen.normal(0, 1, size=1+X.shape[1])
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

    def net_input(self):
        pass 

    def predict(self, X):
        pass
        



