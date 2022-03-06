#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  5 16:47:31 2022

@author: thor
"""
from tabulate import tabulate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import svd

filename="/Users/thor/Git/DTUML/data.csv"

df5=pd.read_csv(filename)
del df5['famhist']
df5.drop('row.names',1,inplace=True)
df5.head()
df4=df2
del df4['famhist']
del df4['chd']

n_df5=(df5-df5.mean())/df5.std()
n_df5.to_csv('ndf5.csv')

### new try

df6=pd.read_csv(filename)

df6.drop('row.names',1,inplace=True)

#n_df6=(df5-df5.mean())/df5.std()
df6.to_csv('df6.csv')
sns.histplot(x="famhist", hue="chd",multiple="dodge",shrink=.8,data=df6)
df6_cov=df6.cov()
sns.heatmap(df6_cov, annot=True, vmin=-70, vmax=70,cmap='RdBu_r')
###

df5.to_csv('df5.csv')
sns.boxplot(data=df5)
headers5=list(df5.columns)
#
df.mean()
normalized_df=(df4-df4.mean())/df4.std()
normalized_df.to_csv("norm.csv")
norm_corr = normalized_df.corr()
sns.pairplot(normalized_df)
M,N,W = svd(normalized_df,full_matrices=False)
var_explained = np.round(N**2/np.sum(N**2), decimals=3)
sns.barplot(x=list(range(1,len(var_explained)+1)), y=var_explained, color="limegreen")
plt.xlabel('SVs', fontsize=16)
plt.ylabel('Percent Variance Explained', fontsize=16)
plt.savefig('svd_scree_plot.png',dpi=100)




#
del df2['row.names']
print(df.isnull().sum())
headers=list(df2.columns)
headers.pop(0)
df2.shape
df2.drop['row.names']
famcount=df['famhist'].value_counts()
chdcount=df['chd'].value_counts()
df2.shape()

corr = df2.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 9))

# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,square=True, linewidths=.5, cbar_kws={"shrink": .5})
print(tabulate(df2.describe(), headers='keys', tablefmt='psql'))

sns.pairplot(df2[['tobacco','typea']])
sns.pairplot(df2[['alcohol','typea']])
sns.pairplot(df2[['alcohol','tobacco']])
sns.pairplot(df2)

