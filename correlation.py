import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

FIG_SIZE=(20, 6)

def heatmap(df):
    fig, ax = plt.subplots(1,3, figsize=FIG_SIZE)

    s1 = sns.heatmap(df.corr('pearson'), ax=ax[0], annot=True)
    s1.set(xlabel='Pearson')

    s2 = sns.heatmap(df.corr('spearman'), ax=ax[1], annot=True)
    s2.set(xlabel='Spearman')

    s3 = sns.heatmap(df.corr('kendall'), ax=ax[2], annot=True)
    s3.set(xlabel='Kendall')


def multi(df, feature):
    fig, ax = plt.subplots(1,3, figsize=FIG_SIZE)

    # when the Pearson and Spearman values are not much different, 
    # data tends to not have extreme values (outliers)
    pearson_corr = df.corr('pearson')[[feature]].sort_values(by=feature, ascending=False)
    spearman_corr = df.corr('spearman')[[feature]].sort_values(by=feature, ascending=False)
    
    # ordinal correlation (Spearman & Kendall Tau)
    kendall_corr = df.corr('kendall')[[feature]].sort_values(by=feature, ascending=False)
    
    s1 = sns.heatmap(pearson_corr, ax=ax[0], annot=True)
    s1.set(xlabel='Pearson')
    s2 = sns.heatmap(spearman_corr, ax=ax[1], annot=True)
    s2.set(xlabel='Spearman')
    s3 = sns.heatmap(kendall_corr, ax=ax[2], annot=True)
    s3.set(xlabel='Kendall')
    
    return pearson_corr, spearman_corr, kendall_corr