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

def generateGPformatDataFun(dfMergeO,VARIABLES):    
# Function to generate data (features and labels) in multi-output GP format
# Returns 
#     X = [N * 2] array, X[:0] = years, and X[:,1] = variable id
#     y = [N * 1] array of variable values
#     varName = [N * 1] array of variable names     
    years = []
    varId = []
    varName = []
    y = []
    for i,key in enumerate(VARIABLES):    
        dfKey = dfMergeO.loc[:,['Year',key]]
        dfKey = dfKey.dropna(axis=0)
        years.extend(dfKey['Year'].values)
        varId.extend(i*np.ones(dfKey['Year'].shape[0],))
        varName.extend(np.array([key for j in np.arange(dfKey['Year'].shape[0])]))
        y.extend(dfKey[key].values)
    years = np.array(years)
    varId = np.array(varId)    
    varName = np.array(varName)    
    y = np.array(y)    
    N = years.shape[0]
    X = np.concatenate((years.reshape(N,1),varId.reshape(N,1)),axis=1)
    y = y.reshape(N,1)        
    return X,y,varName

def generateGPformatXtestFun(X,y,varName):
# Function to generate testing data (features and labels) in multi-output GP format
# Returns 
#     Xtest = [Ntest * 2] array, Xtest[:0] = years, and Xtest[:,1] = variable id
#     yTruth = [Ntest * 1] array of variable values, where truth is available
#     varNameTest = [Ntest * 1] array of variable names     
    xMin = min(X[:,0])-3
    xMax = max(X[:,0])+3    
    NyearsTest = int(xMax+1-xMin)
    yTruth = []
    yearsTest = []
    varIdTest = []
    varNameTest = []
    for i in range(len(np.unique(X[:,1]))):     
        # Ytruth where available
        x_var = X[np.nonzero(X[:, 1]==i), 0].flatten()
        y_var = y[np.nonzero(X[:, 1]==i), 0].flatten()
        yTruthAvai = np.NaN*np.ones((NyearsTest,))
        idxYrTruthAvai = [y in x_var for y in np.arange(xMin,xMax+1)]
        yTruthAvai[idxYrTruthAvai] = y_var
        #
        yTruth.extend(yTruthAvai)
        yearsTest.extend(np.arange(xMin,xMax+1))
        varIdTest.extend(i*np.ones((NyearsTest,)))    
        varNameTest.extend(np.array([varName[i] for j in np.arange(NyearsTest)]))
    yTruth = np.array(yTruth)
    yearsTest = np.array(yearsTest)
    varIdTest = np.array(varIdTest)    
    varNameTest = np.array(varNameTest)    
    Ntest = yearsTest.shape[0]
    Xtest = np.concatenate((yearsTest.reshape(Ntest,1),varIdTest.reshape(Ntest,1)),axis=1)    
    # yearsTest = [yr for yr in np.unique(yearsTest) if yr not in yearsTrain]    
    varNameTest = varNameTest.reshape((Ntest,1))  
    yTruth = yTruth.reshape((Ntest,1))  
    return Xtest,yTruth,varNameTest

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
    # loop over all variables, i.e. RAstk, Mstk, ...
    for i in range(len(np.unique(Yout['VarId']))):
        idxVar = (Yout['VarId']==i)        
        # Plot mean estimate
        ax.plot(Yout.loc[idxVar,'Year'], Yout.loc[idxVar,'Ymu'],color=colorVec[i],linewidth=1)        
        # Plot 95% confidence interval
        ax.fill_between(Yout.loc[idxVar,'Year'], Yout.loc[idxVar,'YmuMinus'], Yout.loc[idxVar,'YmuPlus'], color=colorVec[i], alpha=.15,label=VARIABLES_FULL[i])        
        # Index corresponding to training years
        idxVarTrain = ( (Yout['VarId']==i) & (Yout['Year']>=min(yearsTrain)) & (Yout['Year']<=max(yearsTrain)) )       
        # Index corresponding to testing years
        idxVarTestFuture = ( (Yout['VarId']==i) & (Yout['Year']>max(yearsTrain)) )   
        # Plot truth corresponding to training years
        ax.plot(Yout.loc[idxVarTrain,'Year'], Yout.loc[idxVarTrain,'Y'],'.',color=colorVec[i],markersize=10)
        # Plot truth corresponding to testing years
        ax.plot(Yout.loc[idxVarTestFuture,'Year'], Yout.loc[idxVarTestFuture,'Y'],'x',color=colorVec[i],markersize=9)        
    # Plot training/testing seperation vertical line
    ax.plot([YEAR_TEST-.5,YEAR_TEST-.5],[Yout['YmuMinus'].min(),Yout['YmuPlus'].max()],':k')
    ax.set_xlim([xMin,xMax])   
    ax.set_xlabel('Year')    
    fig.legend()    
    fig.tight_layout()
    return True