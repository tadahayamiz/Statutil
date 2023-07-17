# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:30:27 2020

single rank product test
inspired from:  
FEBS Lett. 2010 March 5; 584(5): 941–944. doi:10.1016/j.febslet.2010.01.031.  

@author: tadahaya
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.stats.multitest as multitest

def calc(data,alpha:float=0.05,correction:str="fdr_bh"):
    """
    conduct the test
    
    Parameters
    ----------
    data: dataframe
        feature x sample matrix

    alpha: float
        indicate the alpha value for the statistical test

    correction: str
        indicate method for correcting multiple tests
        depend on "statsmodels.stats.multitest.multipletests"
    
    """
    dat = Calc()
    dat.set_data(data)
    return dat.calc(alpha,correction)


class Calc():
    def __init__(self):
        self.raw_data = pd.DataFrame()
        self.stat = pd.DataFrame()
        self.index = []
        self.columns = []
        self.n = 0
        self.k = 0
        self.result = pd.DataFrame()


    def set_data(self,data):
        """
        set a dataframe
        
        Parameters
        ----------
        data: dataframe
            feature x sample matrix

        """
        self.raw_data = data.dropna()
        self.index = list(data.index)
        self.columns = list(data.columns)
        self.result = tuple()
        self.n = len(self.index)
        self.k = len(self.columns)
        ori = data.shape
        new = self.raw_data.shape
        if ori!=new:
            print("--- data contains NANs ---")
            print("NANs were deleted: {0} to {1}".format(ori,new))


    def calc(self,alpha:float=0.05,correction:str="fdr_bh"):
        """
        conduct the test
        
        Parameters
        ----------
        alpha: float
            indicate the alpha value for the statistical test

        correction: str
            indicate method for correcting multiple tests
            depend on "statsmodels.stats.multitest.multipletests"
        
        """
        if len(self.index)==0:
            raise ValueError("!! Set a dataframe !!")
        stat = self.__calc_rp()
        pval = stats.gamma.sf(stat.values,self.k,scale=1)
        qval = multitest.multipletests(pval,alpha=alpha,method=correction)[1]
        self.result = pd.DataFrame(self.stat,columns=["log rank product statistic"])
        self.result["p value"] = pval
        self.result["adjusted p value"] = qval
        self.result = self.result.sort_values(by="p value")
        return self.result


    def __calc_rp(self):
        """
        calculate rank product
        the ranking in ascending order (1 is of the highest importance)
        
        """
        rank = self.raw_data.rank(ascending=False)
        rank = np.log(rank)
        self.stat = np.sum(rank,axis=1)*(-1) + self.k * np.log(self.n + 1)
        return self.stat