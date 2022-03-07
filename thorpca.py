#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 14:47:05 2022

@author: thor
"""
from tabulate import tabulate
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

filename="/Users/thor/Git/DTUML/data.csv"
df6=pd.read_csv(filename)

df6.drop('row.names',1,inplace=True)
n_target=df6['chd']
df6.drop(['chd'],inplace=True, axis=1)
df6.drop(['famhist'],inplace=True, axis=1)


# PCA in sklearn

scaler=StandardScaler()
features = ['sbp', 'tobacco', 'ldl', 'adiposity','typea','obesity','alcohol','age']
x = df6.loc[:, features].values
sx_fit=scaler.fit(x)
sx_f_trans=scaler.fit_transform(x)
sns.boxplot(data=sx_f_trans)

pca_m=PCA(n_components=2)
p_cres=pca_m.fit_transform(sx_f_trans)
plt.xlabel("PC2")
plt.ylabel("PC1")
plt.scatter(p_cres[:,0],p_cres[:,1],c=n_target)


df_c=pd.DataFrame(pca_m.components_, index=['pc1','pc2'],columns=df6.columns)
sns.heatmap(df_c,annot=True)          
            
np.sum(pca_m.explained_variance_ratio_)      
 
