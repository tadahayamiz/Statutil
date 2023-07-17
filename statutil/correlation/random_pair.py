# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:30:27 2020

@author: tadahaya
"""

import numpy as np
import pandas as pd
from scipy import stats
from tqdm import trange
from statsmodels.stats.multitest import multipletests


def calc(input:str,data:pd.DataFrame,adjucency:bool=False,target:list=[],perm:int=10,
         spearman:bool=False,correction:str="fdr_bh"):
    """
    calculate p value of the correlation between the input and targets
    
    Parameters
    ----------
    input: str
        indicate the key for the focused sample

    data: dataframe
        sample x feature

    adjucency: bool
        whether data is given adjucency or not

    target: list
        indicate a list of the targets compared with the input

    perm: int
        indicate the iteration counts

    spearman: bool
        whether correlation is Spearman's or Pearson's

    correction: str
        indicate the correction method for multiple correction
        based on statsmodesl.multitest.multipletests
    """        
    dat = Calc()
    dat.set_data(data,adjucency)
    dat.generate_null(perm,spearman)
    if type(input)==str:
        res = dat.calc(input,target,correction)
    elif (type(input)==list) | (type(input)==type(np.array([]))):
        res = dat.calc_from_value(input,correction)
    else:
        raise TypeError("!! Wrong input !!")
    return res,dat


class Calc:
    def __init__(self):
        self.mean = None
        self.var = None
        self.data = pd.DataFrame()
        self.null = pd.DataFrame()
        self.adjucency = None
        self.spearman = None
        

    def set_data(self,data:pd.DataFrame,adjucency:bool=False):
        """
        set a dataframe

        Parameters
        ----------
        data: dataframe
            sample x feature

        adjucency: bool
            whether data is given adjucency or not
        
        """
        self.data = data
        self.adjucency = adjucency


    def random_split(self,obj:list,n:int=2):
        """
        generate random groups from a list
        
        Parameters
        ----------
        obj: list
            a list to be split

        n: int
            number of the elements per group
        
        rem_drop: bool
            whether the remainder is excluded or not

        """
        temp = obj.copy()
        np.random.shuffle(temp)
        n_obj = len(temp)
        d,m = divmod(n_obj,n)
        if m!=0:
            temp = temp[:-m]
            n_obj -= 1
        for idx in range(0,n_obj,n):
            yield temp[idx:idx + n]


    def random_pair_corr(self,df:pd.DataFrame):
        """
        obtain correlation of random pairs from a dataframe
        
        Parameters
        ----------
        df: dataframe
            a df to be calculated
            sample x feature

        """
        temp = df.values
        idx = list(df.index)
        val = list(range(len(idx)))
        pairs = list(self.random_split(val,2))
        corr = []
        ap = corr.append
        if self.adjucency:
            for p in pairs:
                p0 = p[0]
                p1 = p[1]
                ap(temp[p0,p1])
        else:
            for p in pairs:
                p0 = p[0]
                p1 = p[1]
                ap(np.corrcoef(temp[p0],temp[p1])[0,1])
        return corr


    def generate_null(self,perm:int=10,spearman:bool=False):
        """
        generate null distribution
        
        Parameters
        ----------
        perm: int
            indicate the iteration counts

        spearman: bool
            whether correlation is Spearman's or Pearson's

        Returns
        ----------
        mean: float
            indicate the averaged sample mean value of null distribution

        var: float
            indicate the averaged sample variance value of null distribution

        null: pd.DataFrame
            a dataframe of null distribution

        """
        df = self.data.copy()
        self.spearman = spearman
        if spearman:
            df = df.rank(axis=1)
        res = []
        ap = res.append
        for i in trange(perm):
            ap(self.random_pair_corr(df))
        col = ["pair_{}".format(i) for i in range(len(res[0]))]
        self.null = pd.DataFrame(res,columns=col)        
        mean = np.mean(self.null,axis=1)
        self.mean = np.mean(mean)
        var = np.var(self.null,axis=1,ddof=1) # unbiased
        self.var = np.sum(var)/len(var) # if calculated from std, take care root
        self.std = np.sqrt(self.var)
        return self.mean,self.var,self.null


    def set_null(self,data:pd.DataFrame):
        """
        set precalculated null data
        
        """
        self.null = data


    def calc_p(self,value:float):
        """ calculate p value of the input value based on the null distribution """
        return stats.norm.sf(value,loc=self.mean,scale=self.std)


    def calc(self,input:str,target:list=[],correction:str="fdr_bh"):
        """
        calculate p value of the correlation between the input and targets
        
        Parameters
        ----------
        input: str
            indicate the key for the focused sample

        target: list
            indicate a list of the targets compared with the input

        correction: str
            indicate the correction method for multiple correction
            based on statsmodesl.multitest.multipletests
        
        """
        if len(target)==0:
            target = list(self.data.index)
        target = [v for v in target if v!=input]
        res = []
        if self.adjucency:
            for v in target:
                try:
                    res.append(self.data.at[input,v])
                except KeyError:
                    res.append(np.nan)
        else:
            temp = self.data.T
            if self.spearman:
                temp = temp.rank()
            for v in target:
                val = temp[input].values
                try:
                    res.append(np.corrcoef(val,temp[v].values)[0,1])
                except KeyError:
                    res.append(np.nan)
        pval = [self.calc_p(r) for r in res]
        qval = multipletests(pval,method=correction)[1]
        res = pd.DataFrame({"correlation":res,"p value":pval,"adjusted p value":qval},index=target)
        res = res.sort_values("p value")
        return res


    def calc_from_value(self,input:np.array([]),correction:str="fdr_bh"):
        """
        calculate p value of the correlation between the input and targets
        
        Parameters
        ----------
        input: np.array or list
            indicate the key for the focused sample

        target: list
            indicate a list of the targets compared with the input

        correction: str
            indicate the correction method for multiple correction
            based on statsmodesl.multitest.multipletests
        
        """
        pval = [self.calc_p(i) for i in input]
        qval = multipletests(pval,method=correction)[1]
        res = pd.DataFrame({"correlation":input,"p value":pval,"adjusted p value":qval})
        return res