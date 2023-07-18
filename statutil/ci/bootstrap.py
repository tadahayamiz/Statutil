# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 15:46:32 2019

@author: tadahaya
"""

import numpy as np
from sklearn.utils import resample
from scipy.stats import norm, t

class BootstrapCI:
    """
    inspired by https://www.erikdrysdale.com/bca_python/
    
    """
    def __init__(self, func=None, n_boot:int=2000):
        """
        func:
            a function that generates the statistic to be evaluated

        n_boot: int
            the number of bootstrap sampling
        
        """
        self.n_boot = n_boot
        self.func = func
        self.res = dict()
        self.theta = None
        self.store_theta = None
        self.jn = None
        self.n = None
        self.se = None


    def fit(self, data, **kwargs):
        """
        data: np.array
            input for calculating the statistic
            if data is 2d-array, the shape should be samples x features
        
        **kwargs:
            variable lengths arguments in the function
        
        """
        # check stratification
        stratify = None
        if "stratify" in kwargs:
            stratify = kwargs["stratify"]

        # get the baseline statistic
        self.theta = self.func(data, **kwargs)
        self.store_theta = np.zeros(self.n_boot)
        self.jn = self.calc_jackknife(data, **kwargs)
        self.n = len(data)

        # sampling
        for i in range(self.n_boot):
            samples = self.draw_samp(data, stratify=stratify)
            self.store_theta[i] = self.func(samples, **kwargs)
        self.se = self.store_theta.std()


    def calc_ci(self, method:str="bca", alpha:float=0.05, symmetric:bool=True):
        """ calculates bootstrap Confidence Interval """
        if self.theta is None:
            raise ValueError("!! 'fit' before 'calc_ci' !!")
        self.res = {
            "statistic":self.theta, "ci":None, "n_boot":self.n_boot, "method":method
            }
        if method=="bca":
            self.res["ci"] = self.ci_bca(alpha, symmetric)
        elif method=="quantile":
            self.res["ci"] = self.ci_quantile(alpha, symmetric)
        elif method=="se":
            self.res["ci"] = self.ci_se(alpha, symmetric)
        else:
            self.res["method"] = None
            raise KeyError("!! Choose method from {'bca', 'quantile', 'se'} !!")
        return self.res


    def draw_samp(self, data, stratify=None):
        """
        draws data with replacement
        
        Parameters
        ----------
        stratify:
            array-like of shape (n_samples,) or (n_samples, n_outputs)
            for stratification
        
        """
        data = list(data)
        if stratify is not None:
            out = resample(data, stratify=stratify)
        else:
            out = resample(data)
        return out


    def calc_jackknife(self, data, **kwargs):
        """ calculates the Jacknife statistic """
        tmp = data.copy()
        n = len(data)
        jn = np.zeros(n)
        for i in range(n):
            jn[i] = self.func(tmp[:-1], **kwargs)
            tmp = np.roll(tmp, 1) # rolling array 1 by 1
        assert len(jn)==n
        return jn


    def ci_quantile(self, alpha, symmetric):
        """
        calculates quantile-based bootstrap CI
        robust against skewness compared with SE-based
        but cannot account for the bias
        
        """
        if symmetric:
            return np.quantile(self.store_theta, [alpha/2, 1 - alpha/2])
        else:
            return np.quantile(self.store_theta, alpha)
        
    
    def ci_se(self, alpha, symmetric):
        """
        calculates Standard Error based bootstrap CI
        assuming that the statistic is normally ditributed

        """
        if symmetric:
            qq = t(df=self.n - 1).ppf(1 - alpha/2)
            return np.array([self.theta - self.se * qq, self.theta + self.se * qq])
        else:
            qq = t(df=self.n - 1).ppf(1 - alpha)
            return self.theta - self.se * qq

    
    def ci_bca(self, alpha, symmetric):
        """
        calculates bias corrected and accelerated bootstrap CI
        
        """
        if symmetric:
            ql, qu = norm.ppf(alpha/2), norm.ppf(1 - alpha/2)
        else:
            ql, qu = norm.ppf(alpha), norm.ppf(1 - alpha)

        # Acceleration factor
        if self.jn.std() == 0:
            raise ValueError(
                "!! Failed in calculating Jackknife estimators: check data !!"
                )
        num = np.sum((self.jn.mean() - self.jn) ** 3)
        den = 6 * np.sum((self.jn.mean() - self.jn) ** 2) ** 1.5
        a_hat = num / den

        # Bias correction factor
        z_hat = norm.ppf(np.mean(self.store_theta < self.theta))
        a1 = norm.cdf(z_hat + (z_hat + ql) / (1 - a_hat * (z_hat + ql)))
        a2 = norm.cdf(z_hat + (z_hat + qu) / (1 - a_hat * (z_hat + qu)))

        if symmetric:
            return np.quantile(self.store_theta, [a1, a2])
        else:
            return np.quantile(self.store_theta, a1)