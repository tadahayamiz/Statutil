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
        self.data = None
        self.idx = []
        self.col = []
        self.diff = None
        self.result = None
        self.present_member = []


    def set_data(self, data, treatment:str="", control:str="", is_normalized:bool=False):
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

        is_normalized: bool
            whether the given data is normalized or not
            note the analysis needs normalization

        """
        self.col = list(data.columns)
        self.idx = [v.lower() for v in list(data.index)]
        self.data = data.values
        if is_normalized:
            pass
        else:
            self.data = (self.data - np.mean(self.data, axis=1).reshape(-1, 1)) / np.std(self.data, ddof=0, axis=1).reshape(-1, 1)
            check = np.where(np.abs(self.data)==np.inf, np.nan, self.data)
            check = ~np.isnan(check).any(axis=1)
            self.data = self.data[check]
            self.idx = [v for v, w in zip(self.idx, check) if w]
        self.__tre_col = [i for i, v in enumerate(self.col) if treatment in v]
        self.__con_col = [i for i, v in enumerate(self.col) if control in v]
        self.K = len(self.__tre_col)
        self.L = len(self.__con_col)
        self.N = len(self.idx)
        self.result = None


    def calc(self, group:list=[], sign:bool=True, method:str="fdr_bh"):
        """
        calculate the importance of the given group features or groups between the two conditions
        
        Paramters
        ---------
        group: list or dict
            indicate the members of a focusing group, whose length is greater than 2
            dictionary of groups can be acceptable

        sign: bool
            whether sign is considered in integration or not

        method: str
            indicate the method for multiple test correction
            based on statsmodels.stats.multitest.multipletests

        """
        assert len(group) > 2
        if len(self.idx)==0:
            raise ValueError("!! Set a dataframe !!")
        if self.diff is None:
            self._calc_diff()
        if type(group)==list:
            return self.calc_single(group, sign)
        elif type(group)==dict:
            return self.calc_multi(group, sign, method)
        else:
            raise TypeError("!! group should be a list or a dict !!")


    def calc_single(self, group:list=[], sign:bool=True):
        """
        calculate the importance of the given group between the two conditions
        
        Paramters
        ---------
        group: list
            indicate the members of a focusing group

        sign: bool
            whether sign is considered in integration or not
        
        """
        pval, diff, stat = self._calc_indivisual_p(group)
        if len(pval) == 0:
            raise ValueError(
                "!! p value was not calculated. check the correspondence between the group member and the given index !!"
                )
        value = self._integrate(pval, diff, sign)
        tre = np.tile(self.__tre_col, (self.L, 1)).T.flatten()
        con = np.tile(self.__con_col, self.K)
        tre = [self.col[i] for i in tre]
        con = [self.col[i] for i in con]
        each = pd.DataFrame({
            "p_val":pval.flatten(), # K x L -> val[0], val[1], ..., val[k]
            "diff":diff.flatten(),
            "t_stat":stat.flatten(),
            "treatment":tre,
            "control":con,
        })
        return (self._calc_integrated_p(value), value), each


    def calc_multi(self, group:dict, sign:bool=True, method:str="fdr_bh"):
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
        res_s = []
        res_d = []
        res_t = []
        if sign:
            for v in group.values():
                ps, each = self.calc_single(v, sign)
                res_p.append(ps[0]) # p value
                res_s.append(ps[1]) # corrected negative log sum
                res_d.append(np.mean(each["diff"].values))
                res_t.append(np.mean(each["t_stat"].values * np.sign(each["diff"].values)))
        else:
            for v in group.values():
                ps, each = self.calc_single(v, sign)
                res_p.append(ps[0]) # p value
                res_s.append(ps[1]) # corrected negative log sum
                res_d.append(np.mean(each["diff"].values))
                res_t.append(np.mean(each["t_stat"].values))
        res = pd.DataFrame(
            {"p_val":res_p, "corrected_negative_log_sum":res_s, "mean_diff":res_d, "mean_t":res_t},index=list(group.keys())
            )
        res_posi = res.dropna()
        res_q = multitest.multipletests(res_posi["p_val"].values,method=method)[1]
        res.loc[:, "adjusted_p_val"] = [np.nan] * res.shape[0]
        res.loc[res_posi.index, "adjuste_p_val"] = res_q
        res = res.sort_values("p_val")
        return res[["p_val", "adjusted_p_val", "corrected_negative_log_sum", "mean_diff", "mean_t"]]


    def _calc_diff(self):
        """ calculate the differences between the treatment and the control """
        # prepare subtract tensor: index x treatment x control
        self.diff = np.zeros((self.N, self.K, self.L))
        for i, c in enumerate(self.__con_col):
            self.diff[:, :, i] = self.data[:, self.__tre_col] - self.data[:, c].reshape(-1, 1)


    def _calc_indivisual_p(self, group:list=[]):
        """
        calculate the p values of a focusing group

        Parameters
        ----------
        group: list
            indicate the members of a focusing group
        
        """
        self.present_member = [v for v in group if v in self.idx]
        if len(self.present_member)==0:
            return np.array([]), np.array([]), np.array([])
        else:
            present_idx = [self.idx.index(v) for v in self.present_member]
            pval = np.zeros((self.K, self.L))
            diff = np.zeros((self.K, self.L))
            stat = np.zeros((self.K, self.L))
            for k in range(self.K):
                for l in range(self.L):
                    whole = self.diff[:, k, l]
                    focused = whole[present_idx]
                    t, p = stats.ttest_ind(focused, whole, equal_var=False) # Welch
                    stat[k, l] = t
                    pval[k, l] = p
                    diff[k, l] = np.mean(focused) - np.mean(whole)
            return pval, diff, stat


    def _integrate(self, pval:np.array([]), diff:np.array([]), sign:bool=True):
        """
        integrate p values with Fisher's method

        Parameters
        ----------
        pval, diff: list or array
            indicate the p values and the difference of the focused group between two conditions
        
        sign: bool
            whether sign is considered in integration or not

        """
        logs = -np.log(pval)
        if sign:
            logs = np.abs(logs*np.sign(diff))
        integrated = np.sum(logs)
        # divided by L to correct dependence between controls
        return integrated/self.L # take the average. mean false discovery rate may be better? alpha(L + 1)/2L


    def _calc_integrated_p(self, value:float):
        """
        calculate p value based on Gamma distribution with df=K and scale=1

        Parameters
        ----------
        value: float
            integrated negative log sum
        
        """
        return stats.gamma.sf(value, self.K, scale=1)


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
        pval, diff, stat = self._calc_indivisual_p(group)
        integrated = self._integrate(pval, diff, sign)
        p = self._calc_integrated_p(integrated)
        thresh = stats.gamma(a=self.K, scale=1).isf(alpha)
        xmax = np.max((thresh*1.1,integrated*1.1))
        x = np.linspace(0,xmax)
        y = x[x > integrated]
        gamma_pdf = stats.gamma(a=self.K, scale=1).sf
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
        ax.plot(x, gamma_pdf(x),lw=2,c='grey')
        ax.plot(y, gamma_pdf(y),lw=2,c=color)
        ax.axvline(integrated, ymin=0, ymax=1, color=color, lw=2, linestyle="--", label="integrated (p={:.2})".format(p))
        ax.axvline(thresh, ymin=0, ymax=1, color="lightgrey", lw=2, linestyle="--", label="ref (p={:.2})".format(1 - alpha))
        ax.fill_between(x=y, y1=gamma_pdf(y), y2=-0.02, color=color, alpha=0.5)
        plt.legend(loc="best")
        if len(fileout) > 0:
            plt.savefig(fileout, bbox_inches="tight", dpi=dpi)
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


