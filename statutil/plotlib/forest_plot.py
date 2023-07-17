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
        pass


    def make_ax(self):
        """
        generate axis object
        
        Parameters
        ----------
        group: list
            indicate the members of a focusing group
            
        """
        pass


    def get_ci(self,data,alpha=0.95):
        """ calculate confidence intervals """
        n = len(data)
        mu = np.mean(data)
        var = np.var(data,ddof=1)
        se = np.sqrt(var/n)
        ci = stats.t.interval(alpha,n - 1,loc=mu,scale=se)
        return ci


    def prep_input(self,data:dict,keys:list=[],alpha:float=0.95):
        """
        prepare data for forest plot
        
        Parameters
        ----------
        data: dict
            indicate the data

        keys: list
            indicate the members of a focusing group

        alpha: float
            indicate alpha value for the confidence interval

        """
        if len(keys) > 0:
            samples = []
            nf = []
            for g in keys:
                if g in data.keys():
                    samples.append(g)
                else:
                    nf.append(g)
            if len(nf) > 0:
                print("!! CAUTION: {} are not found in the given data !!".format(nf))
            values = [data[s] for s in samples]
        else:
            samples = list(data.keys())
            values = list(data.values())
        mus = [np.mean(v) for v in values]
        cis = [self.get_ci(v,alpha) for v in values]
        upper = [v[1] for v in cis]
        lower = [v[0] for v in cis]
        umax = np.max(upper)
        lmin = np.min(lower)
        end = np.max((np.abs(umax),np.abs(lmin)))*0.1
        uend = umax + end
        lend = lmin - end







def forest_plot(self,group:list=[],ref:float=None,xlabel="value",alpha=0.95,figsize=(6,4),color="navy",title="",
                markersize=15,linewidth=2,fontsize=14,fileout="",dpi=300,marker_alpha=0.7):
    """
    visualize data with forest plot
    
    Parameters
    ----------
    group: list
        indicate the members of a focusing group
        
    """
    # prep data
    data = self.diff
    if len(group) > 0:
        samples = []
        nf = []
        for g in group:
            if g in data.keys():
                samples.append(g)
            else:
                nf.append(g)
        if len(nf) > 0:
            print("!! CAUTION: {} are not found in the given data !!".format(nf))
        values = [data[s] for s in samples]
    else:
        samples = list(data.keys())
        values = list(data.values())
    mus = [np.mean(v) for v in values]
    cis = [self._get_ci(v,alpha) for v in values]
    upper = [v[1] for v in cis]
    lower = [v[0] for v in cis]
    umax = np.max(upper)
    lmin = np.min(lower)
    end = np.max((np.abs(umax),np.abs(lmin)))*0.1
    uend = umax + end
    lend = lmin - end
    
    ### visualization
    zipped = zip(samples,mus,lower,upper)
    
    if len(figsize) > 0:
        plt.figure(figsize=figsize)
    else:
        plt.figure()
    plt.rcParams["font.size"]=fontsize
    plt.gca().invert_xaxis()
    plt.gca().invert_yaxis()
    plt.xlim(left=lend,right=uend)
    plt.xlabel("{0} ({1}% CI)".format(xlabel,int(alpha*100)))
    if len(title) > 0:        
        plt.title(title)
    for sa,mu,lo,up in zipped:
        plt.plot([lo,up],[sa,sa],linewidth=linewidth,color=color)
        plt.plot([mu],[sa],linestyle="",linewidth=0,color=color,
                    marker='o',markersize=markersize,alpha=marker_alpha)
    if ref is not None:
        plt.vlines(x=ref,ymin=samples[0],ymax=samples[-1],color="grey",linestyle="dashed")
    plt.tight_layout()
    if len(fileout) > 0:
        plt.savefig(fileout,dpi=dpi)
    plt.show()