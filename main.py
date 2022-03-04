# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:30:19 2022

@author: s184361
"""

# Imports the numpy and pandas package, then runs the data_prep code
from data_prep import *
# (requires data structures from ex. 2.1.1)

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import matplotlib.pyplot as plt
from scipy.linalg import svd
# Data attributes to be plotted
# %% Scatter plots
'''
for i in range(0,len(attributeNames)-1):
    for j in range(0,len(attributeNames)-1):
        if i ==j:
            print('pass')
            pass
      

        # Make another more fancy plot that includes legend, class labels, 
        # attribute names, and a title.
        f = figure()
        title('South African Heart Disease')
        
        for c in range(C):
            # select indices belonging to class c:
            class_mask = y_CHD==c
            plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)
        
        legend(classNames_CHD)
        xlabel(attributeNames[i])
        ylabel(attributeNames[j])
        
        # Output result to screen
        show()
'''
#%% Histograms
        
#%% PCA
XX = np.delete(X, [4,9], 1)#only data for PCA
XX=np.array(XX,dtype = np.float)
# Subtract mean value from data
Y = XX - np.ones((N,1))*XX.mean(axis=0)

# PCA by computing SVD of Y
U,S,V = svd(Y,full_matrices=False)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1,len(rho)+1),rho,'x-')
plt.plot(range(1,len(rho)+1),np.cumsum(rho),'o-')
plt.plot([1,len(rho)],[threshold, threshold],'k--')
plt.title('Variance explained by principal components');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend(['Individual','Cumulative','Threshold'])
plt.grid()
plt.show()

#%% Plot PC

V = V.T    

# Project the centered data onto principal component space
Z = Y @ V

# Indices of the principal components to be plotted
i = 0
j = 1

# Plot PCA of the data
f = figure()
title('South African Heart Disease: PCA')
#Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y_CHD==c
    plot(Z[class_mask,i], Z[class_mask,j], 'o', alpha=.5)
legend(classNames)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()