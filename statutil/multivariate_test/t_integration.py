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
from itertools import chain
import statsmodels.stats.multitest as multitest
import matplotlib.pyplot as plt
import seaborn as sns


def calc(data,treatment:str="",control:str="",group:list=[],sign:bool=True,method:str="fdr_bh"):
    """
    calculate the importance of the given group or groups between the two conditions
    
    Paramters
    ---------
    data: dataframe
        should contain the columns indicating groups and values

    treatment: str
        indicate the keyword for the treatment columns

    control: str
        indicate the keyword for the control columns

    group: list or dict
        indicate the members of a focusing group
        dictionary of groups can be acceptable

    sign: bool
        whether sign is considered in integration or not

    method: str
        indicate the method for multiple test correction
        based on statsmodels.stats.multitest.multipletests

    """
    dat = Calc()
    dat.set_data(data,treatment,control)
    return dat.calc(group,sign,method)


class Calc():
    def __init__(self):
        self.raw_data = pd.DataFrame()
        self.idx = []
        self.diff = dict()


    def set_data(self,data,treatment:str="",control:str=""):
        """
        set a dataframe in the matrix form (feature x sample)
        
        Parameters
        ----------
        data: dataframe
            should contain the columns indicating groups and values

        treatment: str
            indicate the keyword for the treatment columns

        control: str
            indicate the keyword for the control columns

        """
        col = list(data.columns)
        idx = list(data.index)
        self.__treatment_col = [v for v in col if treatment in v]
        self.__control_col = [v for v in col if control in v]
        tre = data[self.__treatment_col].T
        con = data[self.__control_col].T
        self.idx = []
        for i in idx:
            v_con = con[i].values
            v_tre = tre[i].values
            v_con = v_con[~np.isnan(v_con)]
            v_tre = v_tre[~np.isnan(v_tre)]
            if np.min((v_tre.shape[0],v_con.shape[0])) >= 3:
                self.idx.append(i)
        self.raw_data = data.loc[self.idx,:]
        self.L = len(self.__control_col)
        if self.L==0:
            raise ValueError("!! No control columns: check control argument and the column names !!")
        if len(self.__treatment_col)==0:
            raise ValueError("!! No treaatment columns: check treatment argument and the column names !!")
        self.result = pd.DataFrame()


    def calc(self,group:list=[],sign:bool=True,method:str="fdr_bh"):
        """
        calculate the importance of the given group features or groups between the two conditions
        
        Paramters
        ---------
        group: list or dict
            indicate the members of a focusing group
            dictionary of groups can be acceptable

        sign: bool
            whether sign is considered in integration or not

        method: str
            indicate the method for multiple test correction
            based on statsmodels.stats.multitest.multipletests

        """
        if len(self.idx)==0:
            raise ValueError("!! Set a dataframe !!")
        if len(self.diff)==0:
            self._calc_diff()
        if type(group)==list:
            return self.calc_single(group,sign)
        elif type(group)==dict:
            return self.calc_multi(group,sign,method)
        else:
            raise TypeError("!! group should be a list or a dict !!")


    def calc_single(self,group:list=[],sign:bool=True):
        """
        calculate the importance of the given group between the two conditions
        
        Paramters
        ---------
        group: list
            indicate the members of a focusing group

        sign: bool
            whether sign is considered in integration or not
        
        """
        pval, diff, tstat, pre = self._calc_indivisual_p(group)
        each = pd.DataFrame(
            {"p value":pval, "difference":diff, "t statistics":tstat}, index=pre
            )
        value = self._integrate(pval, diff, sign)
        # note len(pval) = K (treatment condition)
        return self._calc_integrated_p(value,len(pval)), each


    def calc_multi(self,group:dict,sign:bool=True,method:str="fdr_bh"):
        """
        calculate the importance of the given group between the two conditions
        
        Paramters
        ---------
        group: dict
            indicate a dictionary composed of groups
            each value is the memebers of each key

        sign: bool
            whether sign is considered in integration or not
        
        method: str
            indicate the method for multiple test correction
            based on statsmodels.stats.multitest.multipletests

        """
        res_p = []
        res_d = []
        for v in group.values():
            pval, diff, tstat, pre = self.calc_single(v,sign)
            res_p.append(pval)
            res_d.append(np.mean(diff))
        res = pd.DataFrame({"p value":res_p,"mean difference":res_d},index=list(group.keys()))
        res_posi = res.dropna()
        res_q = multitest.multipletests(res_posi["p value"].values,method=method)[1]
        res_posi["adjusted p value"] = res_q
        res_posi = res_posi.sort_values("p value")
        return res_posi[["p value","adjusted p value","mean difference"]]


    def _calc_diff(self):
        """ calculate the differences between the treatment and the control """
        tre = self.raw_data[self.__treatment_col].values
        res = []
        for c in self.__control_col:
            temp = self.raw_data[c].values
            res.append(tre - np.c_[temp])
        res = np.concatenate(res,axis=1)
        self.diff = dict(zip(self.idx,res))
        for k,v in self.diff.items():
            temp = self.diff[k]
            temp = temp[~np.isnan(temp)]
            self.diff[k] = temp
        self.diff["WHOLE"] = np.array(list(chain.from_iterable(self.diff.values())))


    def _calc_indivisual_p(self,group:list=[]):
        """
        calculate the p values of a focusing group

        Parameters
        ----------
        group: list
            indicate the members of a focusing group
        
        """
        present = [v for v in group if v in self.idx]
        if len(present)==0:
            return np.array([]),np.array([])
        else:
            whole = self.diff["WHOLE"]
            mean_whole = np.mean(whole)
            pval = []
            diff = []
            tstat = []
            for v in present:
                temp = self.diff[v]
                t, p = stats.ttest_ind(temp,whole,equal_var=False)
                pval.append(p)
                tstat.append(t)
                diff.append(np.mean(temp) - mean_whole)
            return np.array(pval), np.array(diff), np.array(tstat), present


    def _integrate(self,pval:np.array([]),diff:np.array([]),sign:bool=True):
        """
        integrate p values with Fisher's method

        Parameters
        ----------
        pval,diff: list or array
            indicate the p values and the difference of the focused group between two conditions
        
        sign: bool
            whether sign is considered in integration or not

        """
        if (len(pval)!=len(diff)) and (sign==True):
            raise ValueError("!! length of p values and differences is different. Or sign should be False !!")
        logs = np.log(pval)
        if sign:
            logs = logs*np.sign(diff)
        integrated = np.sum(logs)*(-1)
        return integrated/self.L # take the average. mean false discovery rate may be better? alpha(L + 1)/2L


    def _calc_integrated_p(self,value:float,K:int):
        """
        calculate p value based on Gamma distribution with df=K and scale=1

        Parameters
        ----------
        value: float
            integrated negative log sum
        
        K: int
            degree of freedom for the Gamma distribution

        """
        return stats.gamma.sf(value,K,scale=1)


    def plot(self,res,focus=5,fileout="",dpi=100,thresh=0.05,xlabel="-logP",ylabel="",title="",
                        color="royalblue",alpha=0.5,height=0.8,fontsize=12,textsize=12,figsize=()):
        """
        visualize a result of enrichment analysis
        
        Parameters
        ----------
        res: dataframe
            a result file of enrichment analysis
            
        """
        if len(res) < focus:
            focus = len(res)
        res = res.sort_values(by="p value")
        res = res.iloc[:focus,:]
        res2 = res[res["adjusted p value"] < thresh]
        val = -np.log10(res["adjusted p value"])
        val = [v for v in val[::-1]]
        val2 = -np.log10(res2["adjusted p value"])
        val2 = val2.tolist() + [np.nan]*(len(res) - len(res2))
        val2 = [v for v in val2[::-1]]
        name = list(res.index)
        name = [v.replace("_"," ") for v in name[::-1]]
        X = list(range(focus))
        if len(figsize) > 0:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        if len(xlabel) > 0:
            ax.set_xlabel(xlabel,fontsize=fontsize)
        if len(ylabel) > 0:
            ax.set_ylabel(ylabel,fontsize=fontsize)
        if len(title) > 0:
            ax.set_title(title,fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.set_yticks([])
        ax.spines["right"].set_color("none")
        ax.spines["bottom"].set_color("none")
        ax.spines["top"].set_color("none")
        ax.barh(X,val,color="lightgrey",alpha=alpha,height=height)
        ax.barh(X,val2,color=color,height=height,alpha=alpha)
        for v,w in zip(name,X):
            ax.text(0.02,w,v,ha="left",va="center",fontsize=textsize)
        if len(fileout) > 0:
            plt.savefig(fileout,bbox_inches="tight",dpi=dpi)
        plt.tight_layout()
        plt.show()


    def plot_gamma(self,group:list=[],sign:bool=False,xlabel="",ylabel="probability",
                   title="",color="royalblue",alpha=0.95,fontsize=14,figsize=(),fileout="",dpi=100):
        """
        visualize a result of enrichment analysis
        
        Parameters
        ----------
        group: list
            indicate the members of a focusing group
            
        """
        pval,diff = self._calc_indivisual_p(group)
        integrated = self._integrate(pval,diff,sign)
        K = len(pval)
        p = self._calc_integrated_p(integrated,K)
        thresh = stats.gamma(a=K,scale=1).isf(alpha)
        xmax = np.max((thresh*1.1,integrated*1.1))
        x = np.linspace(0,xmax)
        y = x[x > integrated]
        gamma_pdf = stats.gamma(a=K,scale=1).sf
        if len(figsize) > 0:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.rcParams["font.size"]=fontsize
        if len(xlabel) > 0:
            ax.set_xlabel(xlabel)
        if len(ylabel) > 0:
            ax.set_ylabel(ylabel)
        if len(title) > 0:
            ax.set_title(title)
        ax.plot(x,gamma_pdf(x),lw=2,c='grey')
        ax.plot(y,gamma_pdf(y),lw=2,c=color)
        ax.axvline(integrated,ymin=0,ymax=1,color=color,lw=2,linestyle="--",label="integrated (p={:.2})".format(p))
        ax.axvline(thresh,ymin=0,ymax=1,color="lightgrey",lw=2,linestyle="--",label="ref (p={:.2})".format(1 - alpha))
        ax.fill_between(x=y,y1=gamma_pdf(y),y2=-0.02,color=color,alpha=0.5)
        plt.legend(loc="best")
        if len(fileout) > 0:
            plt.savefig(fileout,bbox_inches="tight",dpi=dpi)
        plt.tight_layout()
        plt.show()


    def _get_ci(self,data,alpha=0.95):
        """ calculate confidence intervals """
        n = len(data)
        mu = np.mean(data)
        var = np.var(data,ddof=1)
        se = np.sqrt(var/n)
        ci = stats.t.interval(alpha,n - 1,loc=mu,scale=se)
        return ci


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


    # under construction
    def plot_diff(self,group:list=[],fileout="",dpi=100,thresh=0.05,xlabel="",ylabel="diff",title="",
                        color="royalblue",alpha=0.5,height=0.8,fontsize=18,figsize=()):
        """
        visualize a result of enrichment analysis
        
        Parameters
        ----------
        res: dataframe
            a result file of enrichment analysis
            
        """
        raise NotImplementedError
        whole = []
        focused = []
        for k,v in self.diff.items():
            temp = list(v)
            whole += temp
            if k in group:
                focused += temp
            else:
                focused += [np.nan]*len(temp)
        temp = np.abs(pd.DataFrame({"whole":whole,"focused":focused}))
        temp = temp.sort_values("whole")
        temp = temp.reset_index()
        if len(figsize) > 0:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()
        ax = fig.add_subplot(111)
        if len(xlabel) > 0:
            ax.set_xlabel(xlabel,fontsize=fontsize)
        if len(ylabel) > 0:
            ax.set_ylabel(ylabel,fontsize=fontsize)
        if len(title) > 0:
            ax.set_title(title,fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        ax.scatter(list(temp.index),list(temp["whole"]),color="lightgrey")
        ax.scatter(list(temp.index),list(temp["focused"]),color=color)
        # sns.distplot(list(temp["whole"]),color="lightgrey",ax=ax)
        # sns.distplot(list(temp["focused"]),color=color,ax=ax)
        if len(fileout) > 0:
            plt.savefig(fileout,bbox_inches="tight",dpi=dpi)
        plt.tight_layout()
        plt.show()


