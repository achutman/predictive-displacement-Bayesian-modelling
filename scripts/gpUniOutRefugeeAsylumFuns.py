# -*- coding: utf-8 -*-
"""
Created on Thu 18 March 2021 

@author: Achut Manandhar, Data Scientist, Migration and Displacement Initiative, Save the Children International
Contact: William Low, Project Lead, Migration and Displacement Initiative, Save the Children International

This software is based on GPy toolbox. In particular, please refer to the following links:
https://github.com/SheffieldML/notebook/blob/master/GPy/multiple%20outputs.ipynb
https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/coregionalized_regression_tutorial.ipynb
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy
import xlrd
import pickle
from sklearn.preprocessing import StandardScaler

def generateGPformatXtestFun(X,y):    
    xMin = min(X[:,0])-3
    xMax = max(X[:,0])+5    
    NyearsTest = int(xMax+1-xMin)    
    # Ytruth where available
    x_var = np.arange(xMin,xMax+1).reshape(NyearsTest,1)
    y_var = np.NaN*np.zeros((NyearsTest,1))    
    idxYrTruthAvai = ((x_var>=X[0])&(x_var<=X[-1])).reshape(NyearsTest,)
    y_var[idxYrTruthAvai] = y         
    return x_var,y_var

def zmuvInvTransformFun(zmuv,yTruth,yPredMu,yPredMuMinus,yPredMuPlus,N_VAR):
   # If standardized, inverse transform to revert back to original scale for plotting and computing errors

   #  Parameters
   #  ----------
   #  zmuv : TYPE
   #      DESCRIPTION.
   #  yPredMu : TYPE
   #      DESCRIPTION.
   #  yPredMuMinus : TYPE
   #      DESCRIPTION.
   #  yPredMuPlus : TYPE
   #      DESCRIPTION.
   #  N_VAR : TYPE
   #      DESCRIPTION.

   #  Returns
   #  -------
   #  None.
    Nsamples = yPredMu.shape[0]
    yTruth = zmuv.inverse_transform(yTruth.reshape(N_VAR,int(Nsamples/N_VAR)).T).T.reshape(Nsamples,1)
    yPredMu = zmuv.inverse_transform(yPredMu.reshape(N_VAR,int(Nsamples/N_VAR)).T).T.reshape(Nsamples,1)
    yPredMuMinus = zmuv.inverse_transform(yPredMuMinus.reshape(N_VAR,int(Nsamples/N_VAR)).T).T.reshape(Nsamples,1)
    yPredMuPlus = zmuv.inverse_transform(yPredMuPlus.reshape(N_VAR,int(Nsamples/N_VAR)).T).T.reshape(Nsamples,1)    
    # Not sure how to inverse transform variance! The following is potentially incorrect!!!
    # yPredVar = zmuv.inverse_transform(yPredVar.reshape(N_VAR,int(Nsamples/N_VAR)).T).T.reshape(Nsamples,1)    
    return yTruth,yPredMu,yPredMuMinus,yPredMuPlus

def plotDataFun(dfMergeO,VARIABLES,VARIABLES_FULL):
    # Plot all time-series datasets    
    xMin = min(dfMergeO['Year'])-3
    xMax = max(dfMergeO['Year'])+3
    markers = ['bo', 'cs', 'rd', 'kP', 'mx', 'g<']
    fig, ax = plt.subplots()        
    for i,key in enumerate(VARIABLES):
        ax.plot(dfMergeO['Year'],dfMergeO[key],markers[i])     
    # ax.set_title('%s'%cnty)
    ax.set_xlabel('Year')
    # ax.set_ylabel('...')
    ax.set_xlim([xMin,xMax])
    ax.legend(VARIABLES_FULL)
    return True

def plotOutputsFun(Yout,yearsTrain,YEAR_TEST):
    colorVec = ['b', 'c', 'r', 'k', 'm', 'g']
    xMin = Yout['Year'].min()
    xMax = Yout['Year'].max()
    fig, ax = plt.subplots()
    # Plot mean estimate
    ax.plot(Yout.loc[:,'Year'], Yout.loc[:,'Ymu'],color='b',linewidth=1)   
    # Plot 95% confidence interval
    ax.fill_between(Yout.loc[:,'Year'], Yout.loc[:,'YmuMinus'], Yout.loc[:,'YmuPlus'], color='b', alpha=.15,label=VARIABLES_FULL[0])    
    # Index corresponding to training years
    idxTrain = ( (Yout['Year'].values>=min(yearsTrain)) & (Yout['Year'].values<=max(yearsTrain)) )       
    # Index corresponding to testing years
    idxTestFuture = ( (Yout['Year'].values>max(yearsTrain)) )   
    # Plot truth corresponding to training years
    ax.plot(Yout.loc[idxTrain,'Year'], Yout.loc[idxTrain,'Y'],'.',color=colorVec[0],markersize=10)
    # Plot truth corresponding to testing years
    ax.plot(Yout.loc[idxTestFuture,'Year'], Yout.loc[idxTestFuture,'Y'],'x',color=colorVec[0],markersize=9)    
    # Plot training/testing seperation vertical line
    ax.plot([YEAR_TEST-.5,YEAR_TEST-.5],[Yout['YmuMinus'].min(),Yout['YmuPlus'].max()],':k')
    ax.set_xlim([xMin,xMax])   
    # plt.ylim([0,5])   
    ax.set_xlabel('Year')
    # plt.ylabel('Refugee & Asylum Stock (10K)')
    # plt.title('%s - RAstk(blue)'%cnty)    
    fig.legend()    
    fig.tight_layout()    
    return True