"""Purpose of this module is to make non parametric estimates of data via sampling
with replacement
Author: Rajan Subramanian
Created: July 15, 2020
"""
import numpy as np
from typing import Union
import matplotlib.pyplot as plt 
import pandas as pd  

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
    - A implementation of empirical bootstrap for estimating paramaters is given
    - A implementation of residual regression bootstrap is given namely: 
        - Get predicted response yhat_i from full data using regression
        - Select bootstrap samples of residuals e_i, where i ranges from 1 to n
        - Calculate bootstrapped yi values by computing yhat_i + e_i
        - Regress bootstrapped yi values on fixed X values to obtain bootstrapped
            regression coefficients
    """
    class _SubSample:
        """Class performs random sample with replacement for each bootstap 
            sample from population data

            Args: 
            pop_sample:     the population sample
            sample_size:    the size of the subsample, if None, uses the length of data
            bsize:          the number of bootstrap subsamples to create

            Returns: 
            iterator of population subsample

        """ 
        def __init__(self, pop_sample: np.ndarray, sample_size: np.ndarray, bsize: int):
            self.pop_sample = pop_sample
            self.sample_size = sample_size 
            self.bsize = bsize
        
        def __iter__(self):
            yield from self.make_sample()
        
        def make_sample(self):
            n = self.pop_sample.shape[0] if self.sample_size is None else self.sample_size 
            for _ in range(self.bsize):
                yield self.pop_sample[np.random.randint(low=0, high=n, size=n)]

    # beginning of Boot definition
    def __init__(self):
        pass 
        
    def empirical_bootstrap(self, pop_data: np.ndarray, n = None, B = 1000, func=None):
        """returns the sample statistic from empirical bootstrap method
        Args:
        pop_data: the data from which we sample with replacement
        n:        the size of the subsample, if None, then uses length of data
        B:        the number of bootstrap subsamples
        func:     the statistc we are interested

        Returns: 
        bootstrapped estimate of the sample statistic
        """
        statistic = []
        for sub_sample in self._SubSample(pop_data, n, B):
            sub_stat = func(sub_sample, axis=0)[0]
            statistic.append(sub_stat)
        mean = np.mean(statistic)
        std_err = np.std(statistic, ddof=1)
        self.statistic = statistic
        self.stat_name = func.__name__
        return (mean, std_err)
    
    def bootstrap_regression(self, X: np.ndarray, n=None, B=1000, model=None):
        """computes standard error from regression model using residual bootstrapping
        Args:
        X:      coefficient matrix, (n_samples, p_features)
                n_samples is number of instances i.e rows
                p_features is number of features i.e columns
              
        n:      the size of the subsample, if None, then use length of data
        B:      the number of bootstrap
        model:  the regression model object after fitting

        Returns: 
        standard error of coefficient estimates
        """
        residuals = model.residuals
        predictions = model.predictions 
        pop_data = np.concatenate((predictions, residuals), axis=1)
        statistic = [None] * B
        self.boot_est = {}  # to store the mean, std_err
        idx = 0   # lcv
        for sub_sample in self._SubSample(pop_data, n, B):
            boot_yi = sub_sample[:, 0] + sub_sample[:, 1]
            model.fit(X, boot_yi)
            statistic[idx] = tuple(model.theta)
            idx += 1
        self.boot_est['mean'] = np.mean(statistic, axis=0)
        self.boot_est['std_err'] = np.std(statistic, ddof=1)
        self.boot_est['sample_statistic'] = statistic 
        self.stat_name = 'residual_bootstrap method'
        return boot_est['std_err']

    def plot_hist(self):
        plt.title(f"""Histogram of Sample {self.stat_name}""")
        plt.hist(self.statistic, orientation='horizontal')

    
        
        