# -*- coding: utf-8 -*-
"""
Created on Fri Nov  9 12:37:31 2018

false discovery rate with Storey method
inspiration from:  
- https://github.com/CancerRxGene/gdsctools//blob/master/doc/index.rst  
- Nature Methods volume 11, pages355–356(2014)  

@author: tadahaya
"""
import numpy as np
import scipy.interpolate


def calc(pval,lambdas:list=[]):
    """
    multiple testing correction with FDR Storey method
    use this for only Storey method
    
    * 181106, some p-values may return increased DEG. Unknown
    
    Parameters
    ----------
    pval: np.array
        an array of p values (,that should be (0,1))
    
    """
    if len(lambdas)!=0:
        dat = QValue(pval,lambdas=lambdas)
    else:
        dat = QValue(pval)
    return dat.qvalue()


class QValue():
    """
    compute Q-value for a given set of P-values with Storey method

    """
    def __init__(self,pval:list, lambdas=None,
        pi0=None, df:int=3, method:str='smoother', 
        smooth_log_pi0:bool=False,verbose:bool=True):
        """
        The q-value of a test measures the proportion of false positives incurred (called the false discovery rate or FDR) 
        when that particular test is called significant.

        Parameters
        ----------
        pval: list or 1d-array
            A vector of p-values (only necessary input)
        
        lambdas: list or 1d-array
            The value of the tuning parameter to estimate pi_0.
            Must be in [0,1). Can be a single value or a range of values.
            If none, the default is a range from 0 to 0.9 with a step size of 0.05 (inluding 0 and 0.9)
            For generation of histgram

        method: str
            Either "smoother" or "bootstrap"; the method for
            automatically choosing tuning parameter in the estimation of
            pi_0, the proportion of true null hypotheses. Only smoother 
            implemented for now.
        
        df: int
            Number of degrees-of-freedom to use when estimating pi_0 with a smoother
            default to 3 i.e., cubic interpolation.

        pi0: float
            if None, it's estimated as suggested in Storey and Tibshirani, 2003.
            May be provided, which is convenient for testing.
            m (=len(pval)) * pi0 tests are hypothesized to be no effects.  
            Thus, m * (1 - lambda) is the estimated No. of true positive under a lambda
            pi0 is unknown, so estimated from lambdas
            pi0(lambda) is 

        smooth_log_pi0: bool
            If True and 'pi0_method' = "smoother",
            pi_0 will be estimated by applying a smoother to a
            scatterplot of log pi_0 rather than just pi_0

        If no options are selected, then the method used to estimate pi_0 is the smoother method described in Storey and Tibshirani (2003).
        The bootstrap method is described in Storey, Taylor & Siegmund (2004) but not implemented yet.

        """
        try:
            self.pval = np.array(pval)
        except:
            self.pval = pval.copy()
        assert(self.pval.min() >= 0 and self.pval.max() <= 1), \
            "p-values should be between 0 and 1"

        if lambdas is None:
            epsilon = 1e-8
            lambdas = scipy.arange(0,0.9 + epsilon,0.05)

        if len(lambdas) > 1 and len(lambdas) < 4:
            raise ValueError("if length of lambda greater than 1, you need at least 4 values")

        if len(lambdas) >= 1 and (min(lambdas) < 0 or max(lambdas) >= 1):
            raise ValueError("lambdas must be in the range[0, 1)")

        self.m = float(len(self.pval))
        self.df = df 
        self.lambdas = lambdas
        self.method = method
        self.verbose = verbose
        self.smooth_log_pi0 = smooth_log_pi0
        self.pi0 = self.estimate_pi0(pi0)


    def estimate_pi0(self,pi0):
        """ Estimate pi0 based on the pvalues """
        pv = self.pval.ravel() # flatten array

        if pi0 is not None:
            pass
        elif len(self.lambdas)==1:
            pi0 = np.mean(pv >= self.lambdas[0])/(1 - self.lambdas[0]) # ratio of true negative
            pi0 = min(pi0,1)
        else:
            # evaluate pi0 for different lambdas
            pi0 = [np.mean(pv >= this)/(1 - this) for this in self.lambdas]
            if self.method=='smoother':
                if (self.smooth_log_pi0):
                    pi0 = np.log(pi0)
                tck = scipy.interpolate.splrep(self.lambdas,pi0,k = self.df)
                pi0 = scipy.interpolate.splev(self.lambdas[-1],tck)
                if self.smooth_log_pi0:
                    pi0 = np.exp(pi0)
                pi0 = min(pi0,1.0) # for assurance of monotonic decrease

            elif self.method == 'bootstrap':
                raise NotImplementedError
                """minpi0 = min(pi0)
                mse = rep(0, len(lambdas))
                pi0.boot = rep(0, len(lambdas))
                for i in range(1,100):
                    p.boot = sample(p, size = m, replace = TRUE)
                    for i in range(0,len(lambdas)):
                        pi0.boot[i] <- mean(p.boot > lambdas[i])/(1 - lambdas[i])
                    mse = mse + (pi0.boot - minpi0)^2

                pi0 = min(pi0[mse == min(mse)])
                pi0 = min(pi0, 1)"""

            if pi0 > 1:
                if self.verbose:
                    print("got pi0 > 1 (%.3f) while estimating qvalues, setting it to 1" % pi0)
                pi0 = 1.0 # pi0 should be less than 1.0
        assert(pi0 >= 0 and pi0 <= 1), "pi0 is not between 0 and 1: %f" % pi0
        return pi0


    def qvalue(self):
        """ Return the qvalues using pvalues stored in pval """
        pv = self.pval.ravel()
        p_ordered = scipy.argsort(pv)
        pv = pv[p_ordered]
        qv = self.pi0 * self.m/len(pv) * pv

        qv[-1] = min(qv[-1],1.0)
        for i in range(len(pv)-2, -1, -1):
            qv[i] = min(self.pi0*self.m*pv[i]/(i + 1.0), qv[i + 1])

        # reorder qvalues
        qv_temp = qv.copy()
        qv = scipy.zeros_like(qv)
        qv[p_ordered] = qv_temp

        # reshape qvalues
        original_shape = self.pval.shape
        qv = qv.reshape(original_shape)
        return qv