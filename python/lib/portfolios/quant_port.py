"""
Implementation of Modern Portfolio Theory
Author: Rajan Subramanian 
Created May 15 2020
"""

import numpy as np
from typing import List

class ModernPortfolioTheory:
    """Implements Harry Markowitz's Model of mean variance optimization
    Args:
    mu_vec:     represents forecasted means of constituents in portfolio
    sigma_vec:  represents forecasted volatility of constituents in portfolio
    cov_mat:    covariance matrix of constituents in portfolio

    Attributes:
    weights:    proportion of capital allocated to security i
    size:       total number of securities in portfolio
    """
    def __init__(self, mu_vec: List[float], sigma_vec: List[float], cov_mat: List[list[float]]):
        self.mu_vec = np.array(mu_vec)
        self.sigma_vec = np.array(sigma_vec)
        self.cov_mat = cov_mat
        self.weights = np.zeros_like(mu_vec)
        self.size = len(mu_vec)

    def compute_portfolio_mean(self) -> float:
        """Returns the portfolio mean: mup = w'mu"""
        return self.weights.T @ self.mu_vec 

    def compute_portfolio_variance(self) -> float:
        """Returns the portfolio variance: w'cov_mat w"""
        return self.weights.T @ self.cov_mat @ self.weights 

    def is_spd(self, x) -> bool:
        """Return True if matrix is spd, else False"""
        return all(np.linalg.eigvals(x) > 0)

    def global_minimum_variance(self, mean_constraint: bool = False, target: None) -> np.array:
        """computes weights associated with global minimum variance
           subject to constraint w'I = 1
           if mean constraint is true, then another constraint is added
           w'mu = target
        """
        n = self.size + 1 if mean_constraint is False else self.size + 2
        A = np.zeros((n,n))
        ones = np.ones(self.size)
        b = np.zeros(n)
        A[:self.size,:self.size] = self.covmat*2
        A[-1,:self.size],A[:self.size,-1] = ones, ones
        if mean_constraint is True:
            A[-2,:self.size],A[:self.size,-2] = self.mu_vec.T,self.mu_vec
            b[-2] = target
        b[-1] = 1