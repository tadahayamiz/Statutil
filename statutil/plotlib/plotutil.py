# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:30:27 2020

Investigate the importance of multivariate in a data set based on Welch's T based integration with negative log sum of p values
inspired from:  doi:10.1186/1471-2105-10-161

@author: tadahaya
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns


class Plot():
    def __init__(self):
        self.figure = None
        self.fontisize = None


    def make_fig(self,figsize=(),title="",xlabel="",ylabel="",fontsize=18):
        """
        generate figure object
        
        Parameters
        ----------
            
        """
        self.fontsize = fontsize
        if len(figsize) > 0:
            self.figure = plt.figure(figsize=figsize)
        else:
            self.figure = plt.figure()
        ax = self.figure.add_subplot(1,1,1)
        if len(title) > 0:
            ax.set_title(title,fontsize=self.fontsize)
        if len(xlabel) > 0:
            ax.set_title(xlabel,fontsize=self.fontsize)
        if len(ylabel) > 0:
            ax.set_title(ylabel,fontsize=self.fontsize)
        return self.figure,ax


    def close_fig(self,fileout:str="",dpi=300,legend:bool=False,loc:str="best",tight:bool=True):
        """
        close figure object
        
        Parameters
        ----------
            
        """
        if tight:
            self.figure.tight_layout()
        if legend:
            self.figure.legend(loc=loc) # 外側を用意しておきたい
        if len(fileout) > 0:
            self.figure.savefig(fileout,dpi=dpi)
        self.figure.show()        


    def make_ax(self):
        """
        generate axis object
        
        Parameters
        ----------
        group: list
            indicate the members of a focusing group
            
        """
        raise NotImplementedError


    def get_ci(self,data,alpha=0.95):
        """ calculate confidence intervals """
        n = len(data)
        mu = np.mean(data)
        var = np.var(data,ddof=1)
        se = np.sqrt(var/n)
        ci = stats.t.interval(alpha,n - 1,loc=mu,scale=se)
        return ci