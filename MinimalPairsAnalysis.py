"""
This script reads a csv with English phoneme confusability data and a csv with 
distinctive features for English phonemes, and then scatterplots and correlates 
the confusability of a contrast with the similarity of that contrast in 
distinctive features.

NOTO BENE!!!!! For this quick and dirty analysis, I apparently just found some
Australian English features! The confusability data are taken from Graff
(2012)'s paper, which themselves were taken from Miller and Nicely (1955).

If I ever went somewhere serious with this, obviously get distinctive features
for American English!

It does appear that confusability is highly correlated with number of features 
in common! I think Pearson R is okay for now -- scatterplot appears to be 
roughly bivariate normally distributed, both X and Y are unbounded, etc.

So next step -- modify Graff/Martin's model with an additional layer for
distinctive features!

"""

import os, itertools
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

os.chdir('/Users/russellrichie/Google Drive/UConn/Classes/DLC (II)/modeling')

"""
Read in confusability data, make a dict of, e.g., confDataDict[frozenset('b','d')] = 3.4
"""
confData = genfromtxt('confusability data.csv', delimiter=',',dtype=str)

confDataDict = dict()
for row in confData[2:]: 
    for colind, letter in enumerate(confData[0][1:]):
        confDataDict[frozenset((letter,row[0]))] = row[colind + 1]

"""
Read in distinctive features data, make a dict of, e.g., distFeatDataDict[frozenset('b','d')] = 1
"""

distFeatData = genfromtxt('english distinctive features.csv', delimiter=',',dtype=str)[1:] #discard the first row, which is just column labels

distFeatDataDict = dict()
for rowPair in itertools.combinations(distFeatData,2): #make a similar dict of distFeatDataDict[frozenset('b','d')] = 3 or whatever the value is
    featDifference = sum(1 for x, y in zip(rowPair[0][1:],rowPair[1][1:]) if x!=y) # rowPair[#][1:]
    distFeatDataDict[frozenset((rowPair[0][0],rowPair[1][0]))] = featDifference

"""
Put the dist feat data and the confusability data together...
"""    
            
allData = []
for confKey, confValue in confDataDict.items():
    try:
        allData.append((confKey,float(confValue),distFeatDataDict[confKey]))
    except:
        continue
"""
...then unzip it so they're each in their own lists again (but sorted!), ready
for plotting and correlation.
"""
listedData = zip(*allData)

labels = listedData[0]
confDataClean = np.array(listedData[1])                
distFeatDataClean = np.array(listedData[2])

plt.xticks(np.arange(.5,5.5,.5), fontsize = 20)
plt.yticks(np.arange(0,12,2), fontsize = 20)

m, b = np.polyfit(confDataClean, distFeatDataClean, 1)

plt.plot(confDataClean, m * confDataClean + b, '-')

plt.show(plt.scatter(confDataClean,distFeatDataClean))
plt.suptitle('More discriminable contrasts differ on more features', fontsize=30)
plt.xlabel('M&N Discriminabiilty score', fontsize=24)
plt.ylabel('Distance in number of features', fontsize=24)

for label, x, y in zip(labels, confDataClean, distFeatDataClean):
    label = list(label)
    label.insert(1,', ')
    plt.annotate(
        ''.join(label), 
        xy = (x, y), xytext = (-20, 20),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        bbox = dict(boxstyle = 'round,pad=0.2', fc = 'yellow', alpha = 0.5),
        arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

print pearsonr(confDataClean,distFeatDataClean)