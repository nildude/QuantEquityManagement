"""Purpose of this module is to make non parametric estimates of data via sampling
with replacement
Author: Rajan Subramanian
Created: July 15, 2020
"""
import numpy as np
from typing import Union 

class Boot:
    """
    Implements the bootstrap methodology via random sampling with replacement
    in order to quantify uncertainty associated with a estimator
    Args:


    Attributes: 

    Notes: 
    Class implements various statistical estimators such as sample mean, 
    sample variance, sample covariance, confidence intervals associated with 
    these estimates and the standard error via random sampling with replacement
    - A implementation of bootstrapped mean and variance is given
    - A implementation of bootstrapped estimates from standard regression is given
    """

    def __init__(self):
        pass 

    def _sub_sample(self, pop_data: np.ndarray, n = None, B = 1000) -> np.ndarray:
        """returns the sample statistic from empirical bootstrap method
        Args:
        pop_data: the data from which we sample with replacement
        n:        the size of the subsample, if None, then uses length of data
        B:        the number of bootstrap subsamples to create in order to estimate the statistic
        func:     the statistc we are interested

        Returns: 
        bootstrapped estimate of the sample statistic
        """
        n = pop_data.shape[0] if n is None else n
        for _ in range(B):
            yield pop_data[np.random.randint(low=0, high=n, size=n)]
        
    def empirical_bootstrap(self, pop_data: np.ndarray, n = None, B = 1000, func=None):
        for boot_sample in self._sub_sample(pop_data=pop_data, n=n, B=B):
            yield func(boot_sample, axis=0)
    
    def get_bootstrap_stats(self, estimates):
        total, n = 0.0, 0
        for x in estimates:
            total += x[0] 
            n += 1 
        return total / n
        
        