# -*- coding: utf-8 -*-
"""
Created on Fri Jul 10 14:30:27 2020

Games-Howell test
inspired from:  
https://www.ijntse.com/upload/1447070311130.pdf
https://aaronschlegel.me/games-howell-post-hoc-multiple-comparisons-test-python.html


@author: tadahaya
"""

import numpy as np
import pandas as pd
from statsmodels.stats.libqsturng import psturng
from itertools import combinations


def calc(data,group_col:str="",val_col:str=""):
    """
    set a dataframe
    
    Parameters
    ----------
    data: dataframe
        should contain the columns indicating groups and values

    group_col: str
        indicates the key for the column representing groups

    val_col: str
        indicates the key for the column representing values

    Returns
    ----------
    p value: dataframe
    t value: dataframe
    corrected degree of freedom: dataframe

    """
    dat = Calc()
    dat.set_data(data,group_col,val_col)
    return dat.calc()


class Calc():
    def __init__(self):
        self.raw_data = pd.DataFrame()
        self.idx = []
        self.mean = dict()
        self.var = dict()
        self.n = dict()
        self.group_col = ""
        self.val_col = ""
        self.p_raw = pd.DataFrame()
        self.p_adj = pd.DataFrame()


    def set_data(self,data,group_col:str="",val_col:str=""):
        """
        set a dataframe
        
        Parameters
        ----------
        data: dataframe
            should contain the columns indicating groups and values

        group_col: str
            indicates the key for the column representing groups

        val_col: str
            indicates the key for the column representing values

        """
        self.raw_data = data
        self.group_col = group_col
        self.val_col = val_col
        self.idx = sorted(list(set(list(data[group_col]))))
        for v in self.idx:
            temp = data[data[group_col]==v]
            self.mean[v] = np.mean(temp[val_col])
            self.var[v] = np.var(temp[val_col],ddof=0) # Note: var should be sample variance
            self.n[v] = temp.shape[0]
        self.result = pd.DataFrame()


    def calc(self):
        """ conduct the test """
        if len(self.idx)==0:
            raise ValueError("!! Set a dataframe !!")
        combs = combinations(self.idx,r=2)
        p_dict = dict()
        t_dict = dict()
        df_dict = dict()
        for cb in combs:
            gr1 = cb[0]
            gr2 = cb[1]
            p,t,df = self.__p(gr1,gr2)
            p_dict[cb] = np.float(p)
            p_dict[(gr2,gr1)] = np.float(p)
            t_dict[cb] = np.float(t)
            t_dict[(gr2,gr1)] = np.float(t)
            df_dict[cb] = int(df)
            df_dict[(gr2,gr1)] = int(df)
        p_res = self.__make_matrix(self.idx,p_dict,fill=1.0)
        t_res = self.__make_matrix(self.idx,t_dict,fill=np.nan)
        df_res = self.__make_matrix(self.idx,df_dict,fill=np.nan)        
        self.result= (p_res,t_res,df_res)
        return self.result


    def __make_matrix(self,idx:list,dic:dict,fill=1.0):
        """ prep matrix """
        res = pd.DataFrame(index=idx,columns=idx)
        for v in idx:
            for w in idx:
                if v==w:
                    res.loc[v,v] = fill
                else:
                    res.loc[v,w] = dic[(v,w)]
        return res


    def __t(self,gr1,gr2):
        """ calculate t value """
        diff = self.mean[gr1] - self.mean[gr2]
        denom = np.sqrt(self.var[gr1]/self.n[gr1] + self.var[gr2]/self.n[gr2])
        return np.abs(diff)/denom


    def __df(self,gr1,gr2):
        """ calculate degree of freedom based on Welch-Satterthwaite equation """
        numer = (self.var[gr1]/self.n[gr1] + self.var[gr2]/self.n[gr2])**2
        denom1 = (self.var[gr1]/self.n[gr1])**2/(self.n[gr1] - 1)
        denom2 = (self.var[gr2]/self.n[gr2])**2/(self.n[gr2] - 1)
        return numer/(denom1 + denom2)


    def __p(self,gr1,gr2):
        """ calculate p value """
        t = self.__t(gr1,gr2)
        df = self.__df(gr1,gr2)
        return psturng(t*np.sqrt(2),len(self.idx),df),t,df
