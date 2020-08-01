"""Linear Classification Models
Author: Rajan Subramanian
Created: May 25, 2020
"""

import numpy as np
from scipy.optimize import minimize
from learning.base import LinearBase
from typing import Union, Dict
from numpy.linalg import norm


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
    def __init__(self, fit_intercept: bool = True, degree: int = 1):
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
        return self.sigm(z) * (1 - self.sigm(z))

    def _jacobian(self,
                  guess: np.ndarray,
                  X: np.ndarray,
                  y: np.ndarray
                  ):
        """Computes the jacobian of likelihood function

        Args:
            guess (np.ndarray): the initial guess for optimizer
            X (np.ndarray): design matrix
                            shape = (n_samples, n_features)
            y (np.ndarray): response variable
                            shape = (n_samples,)

        Returns:
            [np.ndarray]: first partial derivatives wrt weights
        """
        predictions = self.predict(X, guess)
        return -1*(X.T @ (y - predictions))

    def _hessian(self,
                 guess: np.ndarray,
                 X: np.ndarray,
                 y: np.ndarray
                 ):
        """computes the hessian wrt weights

        Args:
            guess (np.ndarray): initial guess for optimizer
            X (np.ndarray): design matrix
                            shape = (n_samples, n_features)
            y (np.ndarray): response variable
                            shape = (n_samples,)

        Returns:
            np.ndarray: second partial derivatives wrt weights
        """
        z = self.net_input(X, thetas=guess)
        prob = self.sigm_prime(z)
        n = prob.shape[0]
        W = np.zeros((n, n))
        np.fill_diagonal(W, prob)
        return X.T @ W @ X

    def _loglikelihood(self, y, z):
        """returns the loglikelihood function of logistic regression

        Args:
            y (np.ndarray): response variable
                            shape = (n_samples,)
            z (np.ndarray): result of net input function
                            shape = (n_samples,)

        Returns:
            np.ndarray: loglikelihood function
        """
        return y @ z - np.log(1 + np.exp(z)).sum()

    def _objective_func(self, guess: np.ndarray, X: np.ndarray, y: np.ndarray):
        """the objective function to be minimized

        Args:
            guess (np.ndarray): initial guess for optimization
            X (np.ndarray): design matrix
                            shape = {n_samples, p_features}
            y (np.ndarray): the response variable
                            shape = {n_samples,}

        Returns:
            float: value from loglikelihood function
        """
        # z = X @ theta
        z = self.net_input(X, thetas=guess)
        f = self._loglikelihood(y, z)
        return -f

    def fit(self,
            X: np.ndarray,
            y: np.ndarray,
            ) -> 'LogisticRegression':
        """fits model to training data and returns regression coefficients
        Args:
        X:
            shape = (n_samples, p_features)
            n_samples is number of instances i.e rows
            p_features is number of features i.e columns
        y:
            shape = (n_samples,)
            Target values

        Returns:
        object
        """
        X = self.make_polynomial(X)
        # generate random guess
        guess_params = np.mean(X, axis=0)
        self.theta = minimize(self._objective_func,
                              guess_params,
                              jac=self._jacobian,
                              hess=self._hessian,
                              method='BFGS',
                              options={'disp': True},
                              args=(X, y))['x']
        return self
    
    def newton_system(self, X, y, func, jac, x, tol_approx=10E-9, tol_consec=10E-6):
        """Solves N-dimensional newton's problem for F(x) = 0

        Args:
            func (function): given function evaluated at x
            jac (function): jacobian of func as a function of x
            x (np.ndarray): initial guess
            tol_approx (float, optional): largest admissible value of ||F(x)||
                                          when solution is found
                                          Defaults to 10E-9.
            tol_consec (float, optional): largest admissible distance between
                                          two consecutive approximations when
                                          solution is found
                                          Defaults to 10E-6.
        """
        xnew, xold = x, x - 1
        count = 0
        while norm(xnew - xold, ord=2) > tol_consec or count > niter:
            xold = xnew
            z = self.net_input(X, thetas=xold)
            predictions = self.predict(X, xold)
            prob = self.sigm_prime(z)
            n = prob.shape[0]
            W = np.zeros((n, n))
            Winv = np.zeros((n, n))
            np.fill_diagonal(W, prob)
            np.fill_diagonal(Winv, 1/prob)
            zbar = X @ xold - Winv @ (y-predictions)
            xnew = np.linalg.inv(X.T @ W @ X) @ X @ W @ zbar
            count += 1
        return xnew

    def net_input(self, X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Computes linear transformation X*theta

        Args:
            X (np.ndarray): design matrix,
            shape = {n_samples, p_features}
            thetas (np.ndarray): weights of logistic function
            shape = {p_features + intercept}

        Returns:
            np.ndarray: linear transformation
        """
        return X @ thetas

    def predict(self,
                X: np.ndarray,
                thetas: np.ndarray = None,
                ) -> Union[np.ndarray, Dict]:
        """Makes predictions of probabilities

        Args:
            X (np.ndarray): design matrix
            shape = {n_samples, p_features}
            thetas (np.ndarray, optional): estimated weights from fitting
            Defaults to None.
            shape = {p_features + intercept,}

        Returns:
            Union[np.ndarray, Dict]: predicted probabilities
            shape = {n_samples,}
        """
        if thetas is None:
            if isinstance(self.theta, np.ndarray):
                return self.sigm(self.net_input(X, self.theta))
            else:
                return self.sigm(self.net_input(X, self.theta['x']))
        return self.sigm(self.net_input(X, thetas))


class LogisticRegressionGD(LinearBase):
    """Implements Logistic Regression via Gradient Descent
       
       Args:
       eta:             Learning rate (between 0.0 and 1.0)
       n_iter:          passees over the training set
       random_state:    Random Number Generator seed
                        for random weight initilization

       Attributes:
       theta:           Weights after fitting
       residuals:       Number of incorrect predictions
    """
    def __init__(self,
                 eta: float = 0.001,
                 n_iter: int = 20,
                 random_state: int = 1,
                 fit_intercept: bool = True,
                 degree: int = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.fit_intercept = fit_intercept
        self.degree = degree
        self.run = False
   
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegressionGD':
        """fits training data
        Args:
        X: shape = {n_samples, p_features}
                    n_samples is number of instances i.e rows
                    p_features is number of features (the dimension of dataset)

        y: shape = {n_samples,}
                    Target values      
        Returns:
        object
        """
        n_samples, p_features = X.shape[0], X.shape[1]
        self.theta = np.zeros(shape=1 + p_features)
        self.cost = []
        X = self.make_polynomial(X)

        for _ in range(self.n_iter):
            z = self.net_input(X, self.theta)
            self.theta += self.eta * self._jacobian(self.theta, X, y)
            self.cost.append(self._cost(y, z) / n_samples)
        self.run = True
        return self
    
    def _jacobian(self,
                  guess: np.ndarray,
                  X: np.ndarray,
                  y: np.ndarray
                  ):
        """Computes the jacobian of likelihood function

        Args:
            guess (np.ndarray): the initial guess for optimizer
            X (np.ndarray): design matrix
                            shape = (n_samples, n_features)
            y (np.ndarray): response variable
                            shape = (n_samples,)

        Returns:
            [np.ndarray]: first partial derivatives wrt weights
        """
        predictions = self.predict(X, guess)
        return (X.T @ (y - predictions))
  
    def _cost(self, y, z):
        """computes cost of likelihood function

        Args:
            y (np.ndarray): response variable
                            shape = (n_samples,)
            z (np.ndarray): result of net input function
                            shape = (n_samples,)

        Returns:
            np.ndarray: loglikelihood function
        """
        # direction is reversed since we are minimizng cost
        return -1 * (y @ z - np.log(1 + np.exp(z)).sum())

    def net_input(self, X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        """Computes linear transformation X*theta

        Args:
            X (np.ndarray): design matrix,
            shape = {n_samples, p_features}
            thetas (np.ndarray): weights of logistic function
            shape = {p_features + intercept}

        Returns:
            np.ndarray: linear transformation
        """
        return X @ thetas
    
    def sigm(self, z: np.ndarray) -> np.ndarray:
        """Computes the sigmoid function

        Args:
            z (np.ndarray): input value from linear transformation

        Returns:
            np.ndarray: sigmoid function value
        """
        return 1.0 / (1 + np.exp(-z))

    def predict(self,
                X: np.ndarray,
                thetas: np.ndarray = None,
                ) -> Union[np.ndarray, Dict]:
        """Makes predictions of probabilities

        Args:
            X (np.ndarray): design matrix
            shape = {n_samples, p_features}
            thetas (np.ndarray, optional): estimated weights from fitting
            Defaults to None.
            shape = {p_features + intercept,}

        Returns:
            Union[np.ndarray, Dict]: predicted probabilities
            shape = {n_samples,}
        """
        if thetas is None:
            if isinstance(self.theta, np.ndarray):
                return self.sigm(self.net_input(X, self.theta))
        return self.sigm(self.net_input(X, thetas))

