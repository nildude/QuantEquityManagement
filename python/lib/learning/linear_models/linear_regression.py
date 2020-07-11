"""Generalized Linear Models
Author: Rajan Subramanian
Created: May 23, 2020
"""
import os
import numpy as np
from learning.base import LinearBase
from scipy.linalg import solve_triangular
from scipy.optimize import minimize
from typing import Dict, Union
import matplotlib.pyplot as plt


class LinearRegression(LinearBase):
    """
    Implements the classic Linear Regression via ols
    Args:
       None

    Attributes:
    theta:           Coefficient Weights after fitting
    errors:          Number of Incorrect Predictions

    Notes:
    Class uses multiple estimation methods to estimate the oridiinary
    lease squares problem min ||Ax - b||, where x = px1, A=nxp, b = nx1
    - A naive implementation of (A'A)^-1 A'b = x is given
      but computing an inverse is expensive
    - A implementation based on QR decomposition is given based on
        min||Ax-b|| = min||Q'(QRx - b)|| = min||(Rx - Q'b)
        based on decomposing nxp matrix A = QR, Q is orthogonal, R is upper triangular
    - A cholesky implementation is also included based on converting an n x p
        into a pxp matrix: A'A = A'b, then letting M = A'A & y = A'b, then we need to 
        solve Mx = y.  Leting M = U'U, we solve this by forward/backward sub
    - A implementation of MLE based on BFGS algorithm is given.  Specifically, we are 
        maximizing log(L(theta)):= L = -n/2 log(2pi * residual_std_error**2) - 0.5 ||Ax-b||
        This is same as minimizing 0.5||Ax-b|| the cost function J.
        The jacobian for regression is given by A'(Ax - b) -> (px1) vector
    - A implementation of MLE based on Newton-CG is provided.  The Hessian is: 
        A'(Ax - b)A -> pxp matrix  
    Todo
    - Levenberg-Marquardt Algorithm

    """

    def __init__(self, fit_intercept: bool=True):
        self.fit_intercept = fit_intercept
        self.theta = None

    def _loglikelihood(self, true, guess):
        error = true - guess
        sigma_error = np.std(error)
        return 0.5 * (error ** 2).sum()

    def _objective_func(self, guess: np.ndarray, A: np.ndarray, b: np.ndarray):
        """the objective function to be minimized, returns estimated x for Ax=b
        Args: 
        guess: 
            initial guess for paramter x
            shape = {1, p_features}
            p_features is the number of columns of design matrix A

        A: 
            the coefficient matrix
            shape = {n_samples, n_features}

        b: 
            the response variable
            shape = {n_samples, 1}

        Returns: 
        Scaler value from loglikelihood function
        """
        y_guess = self.predict(A, thetas=guess)
        f = self._loglikelihood(true=b, guess=y_guess)
        return f

    def _jacobian(self, guess: np.ndarray, A: np.ndarray, b: np.ndarray):
        return (A.T @ (guess @ A.T - b))

    def _hessian(self, guess: np.ndarray, A: np.ndarray, b: np.ndarray):
        return (A.T @ (A @ guess[:, np.newaxis] - b) @ A)
    
    def _levenberg_marqdt(self):
        raise NotImplementedError("Not yet Implemented")

    def estimate_params(self, A: np.ndarray, b: np.ndarray, method: str='ols-qr') -> np.ndarray:
        """numerically solves Ax = b where x is the parameters to be determined
        based on ||Ax - b||
        Args: 
        A: 
            coefficient matrix, (n_samples, n_features)
        b: 
            target values (n_samples, 1)
        """
        if method == 'ols-naive':
            # based on (A'A)^-1 A'b = x
            return np.linalg.inv(A.T @ A) @ A.T @ b
        elif method == 'ols-qr':
            # min||(Rx - Q'b)
            q, r = np.linalg.qr(A)
            # solves by forward substitution
            return solve_triangular(r, q.T @ b)
        elif method == 'ols-cholesky':
            l = np.linalg.cholesky(A.T @ A)
            y = solve_triangular(l, A.T @ b, lower=True)
            return solve_triangular(l.T, y)
        elif method == 'mle-bfgs':
            # generate random guess
            rng = np.random.RandomState(1)
            guess_params = rng.uniform(low=10, high=80, size=A.shape[1])
            # doesn't require hessian
            return minimize(self._objective_func, guess_params, 
                jac=self._jacobian,
                method='BFGS',options={'disp': True}, args=(A,b))
        elif method == 'mle-newton_cg':
            # generate random guess
            rng = np.random.RandomState(1)
            guess_params = rng.uniform(low=10, high=80, size=A.shape[1])
            # hess is optional.  
            return minimize(self._objective_func, guess_params, 
                jac=self._jacobian,hess=self._hessian,
                method='Newton-CG',options={'disp': True}, args=(A,b))
        elif method == 'ols-levenberg-marqdt':
            raise NotImplementedError("Not yet Implemented")

    def fit(self, X: np.ndarray, y: np.ndarray, method: str='ols-cholesky') -> 'LinearRegression':
        """fits training data via ordinary least Squares (ols)
            A wrapper for estimate_params that computes
            regression diagnostics as well
        
        Args:
        X: 
            shape = (n_samples, p_features)
            n_samples is number of instances i.e rows
            p_features is number of features i.e columns
        y: 
            shape = (n_samples)
            Target values

        method: 
            the fitting procedure, default to cholesky decomposition
            options are 'ols-qr','ols', 'mle-bfgs', 'mle_newton_cg'

        Returns:
        object
        """
        
        
        X = self.make_constant(X)
        self.theta = self.estimate_params(A=X, b=y, method=method)
        self.predictions = self.predict(X)
        return self

    def predict(self, X: np.ndarray,
        thetas: Union[np.ndarray, None] = None) -> Union[np.ndarray, Dict]:
        """makes predictions of response variable given input params
        Args:
        X: 
            shape = (n_samples, p_features)
            n_samples is number of instances
            p_features is number of features 
            - if fit_intercept is true, a ones column is needed
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

    def get_residual_diagnostics(self) -> 'LinearRegression':
        """returns the residual diagnostics from fitting process"""

        self.rss = (self.resid**2).sum() 
        self.s2 = self.rss / (n - p)

