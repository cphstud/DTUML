# -*- coding: utf-8 -*-
"""
Created on Fri Mar  4 10:30:19 2022

@author: s184361
"""

# Imports the numpy and pandas package, then runs the data_prep code
from data_prep import *
# (requires data structures from ex. 2.1.1)

from matplotlib.pyplot import figure, plot, title, legend, xlabel, ylabel, show

# Data attributes to be plotted
i = 0
j = 1

# %%
# Make another more fancy plot that includes legend, class labels, 
# attribute names, and a title.
f = figure()
title('South African Heart Disease')

for c in range(C):
    # select indices belonging to class c:
    class_mask = y==c
    plot(X[class_mask,i], X[class_mask,j], 'o',alpha=.3)

legend(classNames)
xlabel(attributeNames[i])
ylabel(attributeNames[j])

# Output result to screen
show()