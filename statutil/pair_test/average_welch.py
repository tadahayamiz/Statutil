# -*- coding: utf-8 -*-
"""
Created on Mon Jan 8 13:05:34 2020

calculate drug-response score based on two group test
Welch's T test statistics
Mann-Whitney's U statistics

MDV o-- Calculator o-- TwoGroupTest

@author: tadahaya
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
import statsmodels.stats.multitest as multitest


def calc(data:pd.DataFrame,key_control:str='',key_treatment:str='',sign:bool=True):
    """
    calculate the difference between the inicated conditions

    Parameters
    ----------
    data: dataframe
        feature x sample dataframe

    key_control, key_treatment: str
        indicator keywords for control and treatment

    """
    dat = AverageWelch()
    res = dat.welch(data,key_control,key_treatment)
    res2 = dat.average(sign=sign)
    return res2,res


class AverageWelch():
    def __init__(self):
        self.res = pd.DataFrame
        self.average_res = pd.DataFrame


    def _calc(self,x1:np.array,x2:np.array):
        """
        scratch Welch's T test for degree of freedom calculation

        Parameters
        ----------
        x1, x2: np.array
            values to be compared
                
        """
        a_mean = np.mean(x1)
        a_var = np.var(x1,ddof=1)
        a_n = len(x1)
        b_mean = np.mean(x2)
        b_var = np.var(x2,ddof=1)
        b_n = len(x2)
        diff = a_mean - b_mean
        s_2 = a_var / a_n + b_var / b_n
        dof = int((a_var / a_n + b_var / b_n)**2 / (((a_var/ a_n)**2 / (a_n-1)) + ((b_var/b_n)**2 / (b_n-1))))
        tval = (a_mean - b_mean) / np.sqrt(s_2)
        l = stats.t.cdf(tval,dof)*2 # two-sided
        h = stats.t.sf(tval,dof)*2
        pval = [x if x==x else 1.0 for x in [l,h]]
        pval = np.min((np.min(pval),1))
        return diff,tval,pval,dof


    def welch(self,data:pd.DataFrame,key_control:str='',key_treatment:str='',correction:str='fdr_bh'):
        """
        calculate the difference between the inicated conditions

        Parameters
        ----------
        data: dataframe
            feature x sample dataframe

        key_control, key_treatment: str
            indicator keywords for control and treatment

        """
        idx = list(data.index)
        p = len(idx)
        col_con = [c for c in list(data.columns) if key_control in c]
        col_tre = [c for c in list(data.columns) if key_treatment in c]
        x_con = data[col_con].values
        x_tre = data[col_tre].values
        res = []
        for i in range(p):
            v_con = x_con[i]
            v_tre = x_tre[i]
            v_con = v_con[~np.isnan(v_con)]
            v_tre = v_tre[~np.isnan(v_tre)]
            if np.min((v_tre.shape[0],v_con.shape[0])) < 3:
                temp = (np.nan,np.nan,np.nan,np.nan)
            else:
                temp = self._calc(v_tre,v_con)
            res.append(temp)
        res = pd.DataFrame(res,index=idx,columns=['difference','statistic','p value','dof'])
        fill = res.dropna()
        if fill.shape[0]==0:
            self.res = pd.DataFrame(columns=['difference','statistic','p value','adjusted p value','dof'])
        else:
            adjp = multitest.multipletests(fill["p value"],alpha=0.05,method=correction)[1]
            fill.loc[:,'adjusted p value'] = adjp.tolist()
            fill = fill.loc[:,['difference','statistic','p value','adjusted p value','dof']]
            null = pd.DataFrame(index=[v for v in list(data.index) if v not in list(fill.index)],columns=fill.columns)
            self.res = pd.concat([fill,null],axis=0,join='inner')
            self.res = self.res.loc[data.index,:]
            self.res = self.res.sort_values('p value')
        return self.res


    def average(self,res:pd.DataFrame=None,sign:bool=True,diff_col:str='difference',stat_col:str='statistic',
                     dof_col:str='dof'):
        """ calculate averaged statistics """
        if res is None:
            res = self.res
        res0 = res.copy().dropna(subset=['statistic'])
        res0 = res0[[diff_col,stat_col,dof_col]]
        if sign==False:
            res0[diff_col] = np.abs(res0[diff_col]).tolist()
            res0[stat_col] = np.abs(res0[stat_col]).tolist()
        res0 = np.mean(res0,axis=0).values.tolist()
        l = stats.t.cdf(res0[1],res0[2])*2 # two-sided
        h = stats.t.sf(res0[1],res0[2])*2
        pval = [x if x==x else 1.0 for x in [l,h]]
        pval = np.min((np.min(pval),1))
        res0.append(pval)
        res0.append(sign)
        self.average_res = pd.DataFrame({'value':res0},index=[diff_col,stat_col,dof_col,'p value','sign']).T
        return self.average_res