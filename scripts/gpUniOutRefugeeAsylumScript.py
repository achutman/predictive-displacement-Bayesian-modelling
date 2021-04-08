# -*- coding: utf-8 -*-
"""
Created on Thu 18 March 2021 

@author: Achut Manandhar, Data Scientist, Migration and Displacement Initiative, Save the Children International
Contact: William Low, Project Lead, Migration and Displacement Initiative, Save the Children International

This work was done in collaboration with University of Virginia
Contact:
Michele Claibourn, DIRECTOR OF SOCIAL, NATURAL, ENGINEERING AND DATA SCIENCES, UVA LIBRARY
David Leblang, PROFESSOR OF POLITICS AND PUBLIC POLICY, DIRECTOR OF THE GLOBAL POLICY CENTER, Frank Batten School of Leadership and Public Policy

This software is based on GPy toolbox. In particular, please refer to the following links:
https://github.com/SheffieldML/notebook/blob/master/GPy/multiple%20outputs.ipynb
https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/coregionalized_regression_tutorial.ipynb

The following script individually models a time-series data related to displacement and predictors of displacement.
It is possible to explore further relevant time-series datasets.
The following processed data will be made available in the HDX website. Please refer to updates in the github page for the corresponding links.
# UNHCR's refugee and asylum seeker stock
# UN DESA's migrant stock (i.e. super set including both all migrants, forced or not)
# UNHCR's refugee and asylum seeker flow
# UNHCR's returned refugees stock
# World Bank's remittance inflow
# UCDP fatalities

Although the key variable being modelled and scored is the UNHCR's refugee and asylum seeker 
stock data, minor modifications can enable modelling and scoring the remaining variables as well.

The script uses modules defined in gpUniOutRefugeeAsylumFuns.py

# Methodolodgy/Algorithm/Pseudocode
gpUniOutRefugeeAsylumScript.py
1. Libraries, parameters, settings
1.1 Import necessary libraries
1.2 Define all parameters and choose various settings
2. Load data, generate and process labels and features
2.1 Read all datasets
2.2 Select countries to model, either manually or based on clustering INFORM risk index
2.3 Loop over selected countries. Learn a multi-output GP model per country.
2.4 Prepare individual time-series datasets before merging 
2.5 (Different compared to multi-output model) No need to merge multiple time-series   
2.6 Replace zero values with NaNs
2.7 Divide dataset into training and testing set (currently, no validation set, but include if optimizing)
2.8 Standardize data based on training set
2.9 Convert labels/features into expected format for GP tool
2.10 Define training/testing sets GP format
3. Train/test
3.1 Define any remaining model parameters
3.2 (Different compared to multi-output model) Define kernels - Add base, bias, white noise.
3.3 Train model
3.4 Generate test data in GP format
3.5 Predict for test data
4. Save/plot outputs
4.1 Save data and forecasts      
4.2 Plot outputs

"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy
import xlrd
import pickle
from sklearn.preprocessing import StandardScaler

cd <please specify the path to gpUniOutRefugeeAsylumFuns.py>

# Specify path to save plots and learned models
# pathSave = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\gaussian process\dataPlots\UNHCR RAstk\timeSepCoregRbfLsFixed3MaxTest2017Nahead1'

# Model parameters
# Select training and testing years
# YEARS_ON = 1981 or 1989 or 1990 (Based on Fearon and Shaver 2020, only consider 1989 onwards). 
# Also, data availability = UNDESA's Migrant Stock >= 1990, World Bank's Remittance >=1980, UCDP >= 1989
YEARS_ON = 1990
# Test years from
YEAR_TEST = 2017
# YEAR_TEST = 2018
# YEAR_TEST = 2019
# For multi-out GP, choose variables to model
VARIABLES = ['RAstk']
VARIABLES_FULL = ['RefAsy Stock']
# Number of variables
N_VAR = len(VARIABLES)
# Specify whether to standardize data. Standardization is recommended.
STANDARDIZED = 1
# Specify save option for plots and learned models
SAVE_OPTION = 0

# Read data
# UNHCR's refugee and asylum seeker stock
pathUnhcr = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UNHCR\processed'
fileName = 'unhcr_refugees_asylums_origin_Iso3.csv'
dfRAstk = pd.read_csv(os.path.join(pathUnhcr,fileName),index_col='Iso3').drop('Country',axis=1)
dfRAstk.columns = dfRAstk.columns.astype(np.int)
# # UN DESA's migrant stock (i.e. super set including both all migrants, forced or not)
# pathUndesa = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UN DESA\Migrant Stock\processed'
# fileName = 'UN_MigrantStockByOrigin.csv'
# dfMstk = pd.read_csv(os.path.join(pathUndesa,fileName))
# # UNHCR's refugee and asylum seeker flow
# fileName = 'unhcr_refugee_asylum_flow_origin_ISO3.csv'
# dfRAflow = pd.read_csv(os.path.join(pathUnhcr,fileName),index_col='Iso3').drop('Country',axis=1)
# dfRAflow.columns = dfRAflow.columns.astype(np.int)
# # UNHCR's returned refugees stock
# fileName = 'unhcr_refugees_returned_origin_Iso3.csv'
# dfRret = pd.read_csv(os.path.join(pathUnhcr,fileName),index_col='Iso3').drop('Country',axis=1)
# dfRret.columns = dfRret.columns.astype(np.int)
# # World Bank's remittance inflow
# pathWB = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\World Bank\processed'
# fileName = 'World Bank Remittance Inflows 2020.csv'
# dfRemit = pd.read_csv(os.path.join(pathWB,fileName),index_col='Iso3',engine='python').drop('Country',axis=1)
# dfRemit.columns = dfRemit.columns.astype(np.int)
# # UCDP fatalities
# pathUcdp = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UCDP\processed'
# fileName = 'ucdp-ged-201-deaths-best-Iso3-2d.csv'
# dfUFat = pd.read_csv(os.path.join(pathUcdp,fileName),index_col='Iso3',engine='python')
# dfUFat.columns = dfUFat.columns.astype(np.int)

# Select countries based on INFORM index. Only consider countries with INFORM risk index >= 5. But the methodology itself applies to any country.
pathInform = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM'
dfInform = pd.read_csv(os.path.join(pathInform,'raw','INFORM2020_TREND_2010_2019_v040_ALL_2 INFORMRiskIndex.csv'))
countries = dfInform.loc[dfInform['2020']>=5,'Iso3'].values
# Skip certain countries for which data is mostly incomplete/missing
# cntySkip = ['PRK','PNG','PSE']
cntySkip = ['PRK','PNG','PSE','SOM']
for cnty in countries:
    if cnty in cntySkip:        
        countries = np.delete(countries,np.where(countries==cnty))
        
# Loop over the remaining countries. Learn a multi-output GP model per country.        
for cnty in countries:        
    print(cnty)        
    # Make a folder to save all plots and learned model per country
    # os.mkdir(os.path.join(pathSave,cnty))
    
    # Prepare individual time-series datasets before merging
    dfRAstkO = dfRAstk.loc[cnty,:]
    dfRAstkO = dfRAstkO.reset_index()
    dfRAstkO.columns = ['Year','RAstk']    
    dfRAstkO = dfRAstkO.loc[dfRAstkO['Year']>=YEARS_ON,:]
    
    # dfRAflowO = dfRAflow.loc[cnty,:]
    # dfRAflowO = dfRAflowO.reset_index()
    # dfRAflowO.columns = ['Year','RAflow']    
    # dfRAflowO = dfRAflowO.loc[dfRAflowO['Year']>=YEARS_ON,:]
    
    # dfMstkO = dfMstk.loc[:,['Year',cnty]]
    # dfMstkO.columns = ['Year','Mstk']    
    
    # dfRemflowO = dfRemflow.loc[cnty,:]
    # dfRemflowO = dfRemflowO.reset_index()
    # dfRemflowO.columns = ['Year','Remflow']    
    # dfRemflowO = dfRemflowO.loc[dfRemflowO['Year']>=YEARS_ON,:]                
    
    # dfRretO = dfRret.loc[cnty,:]
    # dfRretO = dfRretO.reset_index()
    # dfRretO.columns = ['Year','Rret']    
    # dfRretO = dfRretO.loc[dfRretO['Year']>=YEARS_ON,:]
    
    # Merge
    # dfMergeO = dfRAstkO.merge(dfRAflowO,left_on='Year',right_on='Year',how='outer')
    # dfMergeO = dfRAstkO.merge(dfMstkO,left_on='Year',right_on='Year',how='outer')
    # dfMergeO = dfMergeO.merge(dfRretO,left_on='Year',right_on='Year',how='outer')
    
    # Since the data pre-processing codes replaced NaN values with zeros, for now revert/assume zero values to be NaNs. Forcing zeros to be NaNs prevents the model being dominated by zero values.
    dfRAstkO = dfRAstkO.replace(0.,np.NaN)
    
    # Plot all time-series datasets   
    flag = plotDataFun(dfRAstkO,VARIABLES,VARIABLES_FULL)        
    if SAVE_OPTION:
        plt.savefig(os.path.join(pathSave,'%s'%cnty,'data.png'),dpi=300)
    plt.show()
    
    # Define Training/Testing for standardization
    dfRAstkO['Train'] = dfRAstkO['Year']<YEAR_TEST
    dfRAstkO['Test'] = dfRAstkO['Year']>=YEAR_TEST
    
    if STANDARDIZED:
        # Standardize data
        zmuv = StandardScaler()
        zmuv = zmuv.fit(dfRAstkO.loc[dfRAstkO['Train'],VARIABLES].values)
        Xzmuv = zmuv.transform(dfRAstkO.loc[:,VARIABLES].values)
        dfRAstkO.loc[:,VARIABLES] = Xzmuv
        # Plot all time-series datasets after standardizing
        flag = plotDataFun(dfRAstkO,VARIABLES,VARIABLES_FULL)
        if SAVE_OPTION:
            plt.savefig(os.path.join(pathSave,'%s'%cnty,'dataZmuv.png'),dpi=300)
        plt.show()

    # Specify Training/Testing data
    dfRAstkO = dfRAstkO.dropna(axis=0)
    Xtrain = dfRAstkO.loc[dfRAstkO['Train'],['Year']].values
    ytrain = dfRAstkO.loc[dfRAstkO['Train'],VARIABLES].values
    yearsTrain = dfRAstkO.loc[dfRAstkO['Train'],['Year']].values
    
    # Model
    # Define model parameters    
    gpLengthscale = 3.0*max(ytrain)
    
    # # If the model is too confident...
    # # (3) We could return to the intrinsic coregionalization model 
    # # and force the two base covariance functions to share the same coregionalization matrix.
    # kern = GPy.kern.RBF(1, lengthscale=gpLengthscale) + GPy.kern.Bias(1) + GPy.kern.White(1)
    # # kern.name = 'rbf_plus_bias_plus_white'
    # # display(kern)

    # ***(4) Add base, bias, white noise - separate coregionalization matrix
    kern1 = GPy.kern.RBF(1, lengthscale=gpLengthscale)
    kern2 = GPy.kern.Bias(1)
    kern3 = GPy.kern.White(1)    
    kern1.lengthscale.constrain_fixed()
    kern = kern1 + kern2 + kern3
    
    ##############################################################################
    # Fit model
    # model = GPy.models.GPRegression(X, y, kern)
    model = GPy.models.GPRegression(Xtrain, ytrain, kern)
    # model.optimize()
    model.optimize(messages=True)
    model.optimize_restarts(num_restarts = 3)
    if SAVE_OPTION:
        # Save model parameters
        np.save(os.path.join(pathSave,'%s'%cnty,'model_save.npy'), model.param_array)
        # save the model to disk    
        pickle.dump(model, open(os.path.join(pathSave,'%s'%cnty,'model_save.sav'), 'wb'))
        # save zmuv to disk    
        pickle.dump(zmuv, open(os.path.join(pathSave,'%s'%cnty,'zmuv_save.sav'), 'wb'))
    
    # Generate Xtest with Ytruth where available in GP format
    Xtest,yTruth = generateGPformatXtestFun(dfRAstkO.loc[:,['Year']].values,dfRAstkO.loc[:,VARIABLES].values)        
    
    # Predict
    yPredMu, yPredVar = model.predict(Xtest)
    yPredMuMinus = yPredMu-1.96*np.sqrt(yPredVar)
    yPredMuPlus = yPredMu+1.96*np.sqrt(yPredVar)
    
    # If standardized, inverse transform to revert back to original scale for plotting and computing errors
    if STANDARDIZED:
        yTruth,yPredMu,yPredMuMinus,yPredMuPlus = zmuvInvTransformFun(zmuv,yTruth,yPredMu,yPredMuMinus,yPredMuPlus,N_VAR)    
    
    # Generate a dataframe of outputs to save   
    Yout = pd.DataFrame(np.concatenate((Xtest,yTruth,yPredMu,yPredVar,yPredMuMinus,yPredMuPlus),axis=1),columns=['Year','Y','Ymu','Yvar','YmuMinus','YmuPlus'])        
    
    if SAVE_OPTION:
        with pd.ExcelWriter(os.path.join(pathSave,'%s'%cnty,'data_outputs.xlsx')) as writer:
            dfRAstkO.to_excel(writer,sheet_name='data')
            Yout.to_excel(writer,sheet_name='outputs')
    
    # Plot outputs    
    flag = plotOutputsFun(Yout,yearsTrain,YEAR_TEST)    
    if SAVE_OPTION:
        fig.savefig(os.path.join(pathSave,'%s'%cnty,'outputs.png'),dpi=300)
    plt.show()
    
    # # Score
    # # Absolute error
    # Yout['AE'] = abs(Yout['Y'] - Yout['Ymu'])
    # # Absoulte percentage error
    # Yout['APE'] = Yout['AE']/Yout['Ymu']   

