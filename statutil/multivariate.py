# -*- coding: utf-8 -*-
"""
Created on Tue Jul 23 12:09:08 2019

multivariate test

available test:
- T integration

@author: tadahaya
"""
import os
import argparse
import numpy as np
import pandas as pd
import scipy
import csv

from tqdm.auto import trange, tqdm

# original packages in src
from .multivariate_test.t_integration import Calc

### setup ###
if os.name == 'nt':
    SEP = "\\"
elif os.name == 'posix':
    SEP = "/"
else:
    raise ValueError("!! Something wrong in OS detection !!")

parser = argparse.ArgumentParser(description='conduct multivariate test')
parser.add_argument('--note', type=str, help='conduct multivariate test')
parser.add_argument(
    'data',
    type=str,
    help='the path for input data, which is given as a table'
    )
parser.add_argument(
    'group',
    type=str,
    help='the path for input group, which is given as a tsv'
    )
parser.add_argument(
    'control', type=str, default='control',
    help='indicate the keyword for the control columns'
    )
parser.add_argument(
    'treatment', type=str, default='treatment',
    help='indicate the keyword for the control columns'
    )
parser.add_argument(
    '-o', '--outdir',
    type=str, default="",
    help='the path for output'
    )
parser.add_argument(
    '-c', '--correction', default='fdr_bh',
    help='indicates the method for multiple test correction'
    )
parser.add_argument(
    '-s', '--sign', default=True,
    help='whether sign is considered in integration or not'
    )
parser.add_argument(
    '-e', '--extension', default='\t',
    help='indicates the extension of the input files'
    )
parser.add_argument(
    '-v', '--verbose', default=True,
    help='verbose'
    )
parser.add_argument(
    '-n', '--is_normalized', action='store_true',
    help='whether the given data is already normalized or not'
    )


args = parser.parse_args()

### main ###
def ting():
    """
    calculate the importance of the given group or groups between the two conditions
    
    """
    # read group
    with open(args.group) as f:
        reader = csv.reader(f, delimiter=args.extension)
        group = [row[0].lower() for row in reader]
    # read data
    df = pd.read_csv(args.data, sep=args.extension, index_col=0)
    df.index = [v.lower() for v in df.index]
    # check whether group members exist in the indices
    check = [v for v in group if v not in df.index]
    if len(check)!=0:
        print(
            "!! CAUTION: the following members were not found in the given data !!"
            )
        print(check)
    # check whether the indicated keywords exist in the columns of the given data
    con = [v for v in df.columns if args.control in v]
    tre = [v for v in df.columns if args.treatment in v]
    if len(con)==0:
        raise KeyError(f"!! No columns with the indicated control keyword {args.control} !!")
    if len(tre)==0:
        raise KeyError(f"!! No columns with the indicated treatment keyword {args.control} !!")
    if args.verbose:
        print(f"control columns: {con}")
        print(f"treatment columns: {tre}")
    # main
    dat = Calc()
    dat.set_data(df, args.treatment, args.control, args.is_normalized)
    ps, each = dat.calc(group, args.sign, args.correction)
    tmp_diff = each["diff"].mean()
    tmp_stat = each["t_stat"].values.flatten()
    if args.sign:
        tmp_stat = tmp_stat * np.sign(each["diff"].values.flatten())
    res = pd.DataFrame({
        "integrated p value":[ps[0]],
        "corrected negative log sum":[ps[1]],
        "mean difference":tmp_diff,
        "mean t statistics":tmp_stat.mean(),
        "analyzed member":["///".join([v for v in group if v in df.index])],
        "K":[len(tre)], # num of treatment conditions
        "L":[len(con)], # num of control conditions
        }, index=["value"])
    # export
    if len(args.outdir) > 0:
        outdir = args.outdir
    else:
        outdir = os.path.dirname(args.data) + SEP + "result"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    configs = pd.DataFrame({
        "key_control":[args.control],
        "key_treatment":[args.treatment],
        "group":["///".join(group)],
        "sign":args.sign,
        "pandas":[pd.__version__],
        "numpy":[np.__version__],
        "scipy":[scipy.__version__],
        }, index=["value"])
    res.to_csv(outdir + SEP + "result.txt", sep=args.extension)
    each.to_csv(outdir + SEP + "result_individual.txt", sep=args.extension)
    configs.to_csv(outdir + SEP + "config.txt", sep=args.extension)
    # plot
    out_plot = outdir + SEP + "/fig_gamma.tif"
    dat.plot_gamma(group, sign=args.sign, fileout=out_plot)
    

def main():
    raise NotImplementedError


if __name__ == '__main__':
    main()