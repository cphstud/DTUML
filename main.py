# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:30:19 2022

@author: s184361
"""
#from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show
import matplotlib.pyplot as plt
from scipy.linalg import svd
from matplotlib.pyplot import (figure, plot, title, legend, boxplot, xticks, subplot, hist,
                               xlabel,ylabel, ylim, yticks, show)

from scipy.stats import zscore

# Imports the numpy and pandas package, then runs the data_prep code
from data_prep import *
# (requires data structures from ex. 2.1.1)


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
#%% Box plots
# We start with a box plot of each attribute
figure()
title('CHD: Boxplot')
boxplot(X2)
xticks(range(1,M+1), attributeNames, rotation=45)
X3 = np.delete(X, [4,9], 1)#No nominal data
X3=np.array(X3,dtype = np.float)
attributeNames_3 = np.delete(attributeNames, [4,9], 0)
#%%Standardized Box-plot

figure(figsize=(12,6))
title('CHD: Boxplot (standarized)')
boxplot(zscore(X3, ddof=1))
xticks(range(1,M-1), attributeNames_3, rotation=45)

#%% Histograms of all attributes.
figure(figsize=(14,9))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i])
    xlabel(attributeNames[i])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i%v!=0: yticks([])
    if i==0: title('CHD: Histogram')
    
#%% Histograms of all attributes with potential outliers
figure(figsize=(14,9))
m = [0, 1, 2,4,5,6]
for i in range(len(m)):
    subplot(1,len(m),i+1)
    hist(X3[:,m[i]],50)
    xlabel(attributeNames_3[m[i]])
    ylim(0, N) # Make the y-axes equal for improved readability
    if i>0: yticks([])
    if i==0: title('CHD: Histogram (selected attributes)') 


#%% PCA


# Subtract mean value from data
Y = X3 - np.ones((N,1))*X3.mean(axis=0)
Y = Y*(1/np.std(Y,0))#standardize
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
legend(classNames_CHD)
xlabel('PC{0}'.format(i+1))
ylabel('PC{0}'.format(j+1))

# Output result to screen
show()

#%% 

# First 3 PC explain roughly 90%
pcs = np.arange(0,6)
legendStrs = ['PC'+str(e+1) for e in pcs]

bw = .1
r = np.arange(1,M-1)
for i in pcs:    
    plt.bar(r+i*bw, V[:,i], width=bw)
plt.xticks(r+bw, attributeNames_3)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('South African Heart Disease: PCA Component Coefficients')
plt.show()

# Inspecting the plot, we see that the 2nd principal component has large
# (in magnitude) coefficients for attributes sbp, alcohol and age. We can confirm
# this by looking at it's numerical values directly, too:
print('PC2:')
print(V[:,1].T)

#%%