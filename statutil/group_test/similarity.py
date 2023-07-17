# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:30:27 2020

@author: tadahaya
"""

import numpy as np
import pandas as pd
import scipy.stats as stats
from tqdm import trange

from statsmodels.stats.libqsturng import psturng
from itertools import combinations

def calc(data:pd.DataFrame(),group:list=[],**kwargs):
    """
    calculate similarity of the indicated group
    
    Parameters
    ----------
    data: dataframe
        adjucency matrix

    group: list
        indicate the group of interest
        should be the members of the given dataframe

    Returns
    -------
    p value: dataframe
    t value: dataframe
    corrected degree of freedom: dataframe

    """
    dat = Calc()
    dat.set_data(data)
    return dat.calc(group,**kwargs)


# manager
class Calc():
    def __init__(self):
        self.algorithm = RankProduct()


    def to_tintegration(self):
        self.algorithm = TIntegration()


    def to_rankproduct(self):
        self.algorithm = RankProduct()


    def set_data(self,data:pd.DataFrame):
        """
        set a dataframe

        Parameters
        ----------
        data: dataframe
            adjucency matrix of the background
        
        """
        self.algorithm.set_data(data)


    def calc(self,group:list=[],**kwargs):
        """
        calculate p value of group similarity
        
        Parameters
        ----------
        group: list
            indicate the group to be evaluated
        
        """
        return self.algorithm.calc(group,**kwargs)


# deligation
class TIntegration():
    def __init__(self):
        self.statistic = None
        self.p = None
        self.dof = None
        self.diff = None
        self.each_value = pd.DataFrame()
        self.res = pd.DataFrame()


    def set_data(self,data:pd.DataFrame):
        """
        set a dataframe

        Parameters
        ----------
        data: dataframe
            adjucency matrix of the background
        
        """
        self.data = data


    def ttest(self,x1:np.array,x2:np.array):
        """
        scratch Welch's T test for degree of freedom calculation
        one sided, greater

        Parameters
        ----------
        x1,x2: np.array
            values to be compared
                
        """
        a_mean = np.mean(x1)
        a_var = np.var(x1,ddof=1)
        a_n = len(x1)
        b_mean = np.mean(x2)
        b_var = np.var(x2,ddof=1)
        b_n = len(x2)
        s_2 = a_var / a_n + b_var / b_n
        dof = int((a_var / a_n + b_var / b_n)**2 / (((a_var/ a_n)**2 / (a_n-1)) + ((b_var/b_n)**2 / (b_n-1))))
        t_value = (a_mean - b_mean) / np.sqrt(s_2)
        p_value = stats.t.sf(t_value, dof)
        return t_value,p_value,dof


    def average_t(self,t_values:list,dof:list):
        """
        calculate averaged t values
        one sided, greater
        
        Parameters
        ----------
        t_values: list
            t values to be averaged

        dof: list
            degree of freedom corresponding to t values

        """
        t_val = np.mean(t_values)
        dof0 = np.mean(dof)
        return t_val,stats.t.sf(t_val,dof0),dof0
        

    def calc(self,group:list=[]):
        """
        calculate p value of group similarity based on T integration
        
        Parameters
        ----------
        group: list
            indicate the group to be evaluated
        
        """
        temp = self.data.loc[group,:]
        temp2 = temp[group]
        whole = list(self.data.columns)
        res_t = []
        res_p = []
        diff = []
        dof = []
        for g in group:
            other = [v for v in whole if v!=g]
            other2 = [v for v in group if v!=g]
            temp_ = temp.loc[g,other].values.flatten()
            temp2_ = temp2.loc[g,other2].values.flatten()
            t,p,d = self.ttest(temp2_,temp_)
            res_t.append(t)
            res_p.append(p)
            diff.append(np.mean(temp2_) - np.mean(temp_))
            dof.append(d)
        self.statistic,self.p,self.dof = self.average_t(res_t,dof)
        self.diff = np.mean(diff)
        self.each_value = pd.DataFrame({"t":res_t,"p":res_p,"diff":diff,"dof":dof},index=group)
        group_str = _list2str(group)
        temp = [self.statistic,self.p,self.dof,self.diff,group_str]
        idx = ["statistic","p value","dof","diff","group"]
        self.res = pd.DataFrame({"result":temp},index=idx).T
        return self.res


# concrete class
class RankProduct():
    def __init__(self):
        self.statistic = None
        self.p = None
        self.dof = None
        self.diff = None
        self.each_value = pd.DataFrame()
        self.res = pd.DataFrame()


    def set_data(self,data:pd.DataFrame):
        """
        set a dataframe

        Parameters
        ----------
        data: dataframe
            adjucency matrix of the background
        
        """
        self.data = data


    def rankproduct(self,key:str,group:list):
        """
        calculate rank product p values based on the indicated key and list of a group

        Parameters
        ----------
        key: str
            indicate the column of interest

        group: list
            indicate the columns (or indices) of interest

        """
        idx = list(self.data.index)
        other = [v for v in idx if v!=key]
        rank = self.data.loc[other,[key]].rank(ascending=False)
        rank = rank.loc[group]
        rank = np.log(rank).values
        stat = np.sum(rank)*(-1) + len(group) * np.log(len(other) + 1)
        p = stats.gamma.sf(stat,len(group),scale=1)
        return stat,p


    def calc(self,group:list):
        """
        calculate rank product p values based on the indicated list of a group

        Parameters
        ----------
        group: list
            indicate the columns (or indices) of interest

        """
        res_p = []
        res_stat = []
        for g in group:
            temp = [v for v in group if v!=g]
            stat,p = self.rankproduct(g,temp)
            res_stat.append(stat)
            res_p.append(p)
        self.each_value = pd.DataFrame({"statistic":res_stat,"p value":res_p},index=group)
        self.statistic = np.mean(res_stat)
        self.p = stats.gamma.sf(self.statistic,len(group) - 1,scale=1)
        group_str = _list2str(group)
        temp = [self.statistic,self.p,group_str]
        idx = ["statistic","p value","group"]
        self.res = pd.DataFrame({"result":temp},index=idx).T
        return self.res


def _list2str(obj:list,sep:str=";"):
    """ convert list to str combined with the indicated separator """
    res = str(obj[0])
    for v in obj[1:]:
        res += ";{}".format(str(v))
    return res