# -*- coding: utf-8 -*-
"""
Created on Thu 18 March 2021 

@author: Achut Manandhar, Data Scientist, Migration and Displacement Initiative, Save the Children International
Contact: William Low, Project Lead, Migration and Displacement Initiative, Save the Children International

This software is based on GPy toolbox. In particular, please refer to the following links:
https://github.com/SheffieldML/notebook/blob/master/GPy/multiple%20outputs.ipynb
https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/coregionalized_regression_tutorial.ipynb

The following script simultaneously models multiple time-series data related to displacement and predictors of displacement.
It is possible to add further relevant time-series datasets.
The following processed data will be made available in the HDX website. Please refer to updates in the github page for the corresponding links.
# UNHCR's refugee and asylum seeker stock
# UN DESA's migrant stock (i.e. super set including both all migrants, forced or not)
# UNHCR's refugee and asylum seeker flow
# UNHCR's returned refugees stock
# World Bank's remittance inflow
# UCDP fatalities

Although the key variable being scored is the UNHCR's refugee and asylum seeker 
stock data, minor modifications can enable scoring the remaining variables as well.

The script uses modules defined in gpMultiOutRefugeeAsylumFuns.py

# Methodolodgy/Algorithm/Pseudocode
gpMultiOutRefugeeAsylumScript.py
1. Libraries, parameters, settings
1.1 Import necessary libraries
1.2 Define all parameters and choose various settings
2. Load data, generate and process labels and features
2.1 Read all datasets
2.2 Select countries to model, either manually or based on clustering INFORM risk index
2.3 Loop over selected countries. Learn a multi-output GP model per country.
2.4 Prepare individual time-series datasets before merging 
2.5 (Key multi-output model step) Merge multiple time-series   
2.6 Replace zero values with NaNs
2.7 Divide dataset into training and testing set (currently, no validation set, but include if optimizing)
2.8 Standardize data based on training set
2.9 Convert labels/features into expected format for multi-output GP tool
2.10 Define training/testing sets in multi-output GP format
3. Train/test
3.1 Define any remaining model parameters
3.2 (Key multi-output model step) Define kernels - Add base, bias, white noise, define appropriate coregionalization matrix
3.3 Train model
3.4 Generate test data in multi-output GP format
3.5 Predict for test data
4. Save/plot outputs
4.1 Save data and forecasts      
4.2 Plot outputs
    
"""
# (1) Libraries, configuration settings, model parameters
# (1.1) Import necessary libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import GPy
import xlrd
import pickle
from sklearn.preprocessing import StandardScaler   

cd <please specify the path to gpMultiOutRefugeeAsylumFuns.py>

# Specify path to save plots and learned models
# pathSave = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\gaussian process\dataPlots\UNHCR RAstkMstkRretRAflowRemUfat\timeSepCoregRbfLsFixed3MaxTest2017Nahead1'

# (1.2) Define all parameters and choose various settings
# Select training and testing years
# YEARS_ON = 1981 or 1989 or 1990 (Based on Fearon and Shaver 2020, only consider 1989 onwards). 
# Also, data availability = UNDESA's Migrant Stock >= 1990, World Bank's Remittance >=1980, UCDP >= 1989
YEARS_ON = 1990
# Test years from
YEAR_TEST = 2017
# YEAR_TEST = 2018
# YEAR_TEST = 2019
# For multi-out GP, choose variables to model
# VARIABLES = ['RAstk','Mstk']
# ...
VARIABLES = ['RAstk','Mstk','Rret','RAflow','Remit','UFat']
# Variable names in full for plotting
VARIABLES_FULL = ['RefAsy Stock','Migrant Stock','Ref Returnees','RefAsy Flow','Remittance','Fatalities']
# Number of variables
N_VAR = len(VARIABLES)
# Specify whether to standardize data. Standardization is recommended.
STANDARDIZED = 1
# Specify save option for plots and learned models
SAVE_OPTION = 0


# (2) Data load, prepare, process
# (2.1) Read all datasets
# UNHCR's refugee and asylum seeker stock
pathUnhcr = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UNHCR\processed'
fileName = 'unhcr_refugees_asylums_origin_Iso3.csv'
dfRAstk = pd.read_csv(os.path.join(pathUnhcr,fileName),index_col='Iso3').drop('Country',axis=1)
dfRAstk.columns = dfRAstk.columns.astype(np.int)
# UN DESA's migrant stock (i.e. super set including both all migrants, forced or not)
pathUndesa = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UN DESA\Migrant Stock\processed'
fileName = 'UN_MigrantStockByOrigin.csv'
dfMstk = pd.read_csv(os.path.join(pathUndesa,fileName))
# UNHCR's refugee and asylum seeker flow
fileName = 'unhcr_refugee_asylum_flow_origin_ISO3.csv'
dfRAflow = pd.read_csv(os.path.join(pathUnhcr,fileName),index_col='Iso3').drop('Country',axis=1)
dfRAflow.columns = dfRAflow.columns.astype(np.int)
# UNHCR's returned refugees stock
fileName = 'unhcr_refugees_returned_origin_Iso3.csv'
dfRret = pd.read_csv(os.path.join(pathUnhcr,fileName),index_col='Iso3').drop('Country',axis=1)
dfRret.columns = dfRret.columns.astype(np.int)
# World Bank's remittance inflow
pathWB = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\World Bank\processed'
fileName = 'World Bank Remittance Inflows 2020.csv'
dfRemit = pd.read_csv(os.path.join(pathWB,fileName),index_col='Iso3',engine='python').drop('Country',axis=1)
dfRemit.columns = dfRemit.columns.astype(np.int)
# UCDP fatalities
pathUcdp = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\UCDP\processed'
fileName = 'ucdp-ged-201-deaths-best-Iso3-2d.csv'
dfUFat = pd.read_csv(os.path.join(pathUcdp,fileName),index_col='Iso3',engine='python')
dfUFat.columns = dfUFat.columns.astype(np.int)

# (2.2) Select countries to model, either manually or based on INFORM risk index
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
    
    # Generate Labels/features
    # (2.4) Prepare individual time-series datasets before merging
    dfRAstkO = dfRAstk.loc[cnty,:]
    dfRAstkO = dfRAstkO.reset_index()
    dfRAstkO.columns = ['Year','RAstk']    
    dfRAstkO = dfRAstkO.loc[dfRAstkO['Year']>=YEARS_ON,:]
    
    dfMstkO = dfMstk.loc[:,['Year',cnty]]
    dfMstkO.columns = ['Year','Mstk']    
    dfMstkO = dfMstkO.loc[dfMstkO['Year']>=YEARS_ON,:]
    
    dfRretO = dfRret.loc[cnty,:]
    dfRretO = dfRretO.reset_index()
    dfRretO.columns = ['Year','Rret']    
    dfRretO = dfRretO.loc[dfRretO['Year']>=YEARS_ON,:]
    
    dfRAflowO = dfRAflow.loc[cnty,:]
    dfRAflowO = dfRAflowO.reset_index()
    dfRAflowO.columns = ['Year','RAflow']    
    dfRAflowO = dfRAflowO.loc[dfRAflowO['Year']>=YEARS_ON,:]
    
    dfRemitO = dfRemit.loc[cnty,:]
    dfRemitO = dfRemitO.reset_index()
    dfRemitO.columns = ['Year','Remit']    
    dfRemitO = dfRemitO.loc[dfRemitO['Year']>=YEARS_ON,:]  

    dfUFatO = dfUFat.loc[cnty,:]
    dfUFatO = dfUFatO.reset_index()
    dfUFatO.columns = ['Year','UFat']    
    dfUFatO = dfUFatO.loc[dfUFatO['Year']>=YEARS_ON,:]                    
    
    # (2.5) (Key multi-output model step) Merge multiple time-series   
    dfMergeO = dfRAstkO.merge(dfMstkO,left_on='Year',right_on='Year',how='outer')
    dfMergeO = dfMergeO.merge(dfRretO,left_on='Year',right_on='Year',how='outer')
    dfMergeO = dfMergeO.merge(dfRAflowO,left_on='Year',right_on='Year',how='outer')
    dfMergeO = dfMergeO.merge(dfRemitO,left_on='Year',right_on='Year',how='outer')
    dfMergeO = dfMergeO.merge(dfUFatO,left_on='Year',right_on='Year',how='outer')        
    
    # Process labels/features
    # (2.6) Since the data pre-processing codes replaced NaN values with zeros, for now revert/assume zero values to be NaNs. Forcing zeros to be NaNs prevents the model being dominated by zero values.
    dfMergeO = dfMergeO.replace(0.,np.NaN)
    # print(dfMergeO.isnull().all())
    
    # Plot all time-series datasets   
    flag = plotDataFun(dfMergeO,VARIABLES,VARIABLES_FULL)    
    if SAVE_OPTION:
        plt.savefig(os.path.join(pathSave,'%s'%cnty,'data.png'),dpi=300)
    plt.show()
    
    # (2.7) Define Training/Testing for standardization
    dfMergeO['Train'] = dfMergeO['Year']<YEAR_TEST
    dfMergeO['Test'] = dfMergeO['Year']>=YEAR_TEST
    
    if STANDARDIZED:
        # (2.8) Standardize data based on training set
        zmuv = StandardScaler()
        zmuv = zmuv.fit(dfMergeO.loc[dfMergeO['Train'],VARIABLES].values)
        Xzmuv = zmuv.transform(dfMergeO.loc[:,VARIABLES].values)
        dfMergeO.loc[:,VARIABLES] = Xzmuv    
        # Plot all time-series datasets after standardizing
        flag = plotDataFun(dfMergeO,VARIABLES,VARIABLES_FULL)
        if SAVE_OPTION:
            plt.savefig(os.path.join(pathSave,'%s'%cnty,'dataZmuv.png'),dpi=300)
        plt.show()
    
    # (2.9) Convert labels/features into the expected format for multi-out GP tool
    X,y,varName = generateGPformatDataFun(dfMergeO,VARIABLES)
    data = {'X':X,'Y':y}
    dfXY = pd.DataFrame(np.concatenate((X,y),axis=1),columns=['Year','VarId','Y'])
    dfXY['VarName'] = varName    
    
    # Uncomment to debug and print partial data
    # print('First column of X contains the years.')
    # print(np.unique(data['X'][:, 0]))
    # print('Second column of X contains the RAstk')
    # print(np.unique(data['X'][:, 1]))

    # (2.10) Define training/testing sets in multi-output GP format
    Xtrain = X[X[:,0]<YEAR_TEST,:]
    ytrain = y[X[:,0]<YEAR_TEST,:]
    yearsTrain = np.unique(Xtrain[:,0])
    
    # (3) Model training and testing
    # (3.1) Define any remaining model parameters    
    gpLengthscale = 3.0*max(ytrain)
    
    # # If the model is too confident...
    # # (3.2) We could return to the intrinsic coregionalization model 
    # # and force the two base covariance functions to share the same coregionalization matrix.
    # kern1 = GPy.kern.Matern32(1, lengthscale=gpLengthscale) + GPy.kern.Bias(1) + GPy.kern.White(1)
    # kern1.name = 'rbf_plus_bias_plus_white'
    # kern2 = GPy.kern.Coregionalize(1,output_dim=N_VAR+1, rank=5)
    # kern = kern1**kern2
    # # kern.name = 'product'
    # # display(kern)
    
    # (3.2) (Key multi-output model step) 
    # Define kernels - Add base, bias, white noise - separate coregionalization matrix
    # kern1 = GPy.kern.Linear(1)**GPy.kern.Coregionalize(1,output_dim=N_VAR+1, rank=1)
    kern1 = GPy.kern.RBF(1, lengthscale=gpLengthscale)**GPy.kern.Coregionalize(1,output_dim=N_VAR+1, rank=1)
    kern2 = GPy.kern.Bias(1)**GPy.kern.Coregionalize(1,output_dim=N_VAR+1, rank=1)
    kern3 = GPy.kern.White(1)**GPy.kern.Coregionalize(1,output_dim=N_VAR+1, rank=1)
    kern1.rbf.lengthscale.constrain_fixed()
    kern = kern1 + kern2 + kern3
    
    ##############################################################################
    # (3.3) ***Key model training step***
    # model = GPy.models.GPRegression(X, y, kern)
    model = GPy.models.GPRegression(Xtrain, ytrain, kern)    
    model.optimize(messages=True)
    model.optimize_restarts(num_restarts = 3)
    if SAVE_OPTION:
        # Save model parameters
        np.save(os.path.join(pathSave,'%s'%cnty,'model_save.npy'), model.param_array)
        # save the model to disk    
        pickle.dump(model, open(os.path.join(pathSave,'%s'%cnty,'model_save.sav'), 'wb'))
        # save zmuv to disk    
        pickle.dump(zmuv, open(os.path.join(pathSave,'%s'%cnty,'zmuv_save.sav'), 'wb'))
        
        
    # (3.4) Generate Xtest with Ytruth where available in multi-output GP format
    Xtest,yTruth,varNameTest = generateGPformatXtestFun(X,y,dfXY['VarName'].unique())    
    
    # (3.5) ***Key prediction step***
    # Predict
    yPredMu, yPredVar = model.predict(Xtest)
    yPredMuMinus = yPredMu-1.96*np.sqrt(yPredVar)
    yPredMuPlus = yPredMu+1.96*np.sqrt(yPredVar)
    
    # (3.6) If standardized, inverse transform to revert back to original scale for plotting and computing errors
    if STANDARDIZED:
        yTruth,yPredMu,yPredMuMinus,yPredMuPlus = zmuvInvTransformFun(zmuv,yTruth,yPredMu,yPredMuMinus,yPredMuPlus,N_VAR)

    # (4) Save and plot outputs
    # Generate a dataframe of outputs to save    
    Yout = pd.DataFrame(np.concatenate((Xtest,yTruth.astype(np.float),yPredMu,yPredVar,yPredMuMinus,yPredMuPlus),axis=1),columns=['Year','VarId','Y','Ymu','Yvar','YmuMinus','YmuPlus'])    
    Yout['VarName'] = varNameTest        
    
    # (4.1) Save data and forecasts 
    if SAVE_OPTION:
        with pd.ExcelWriter(os.path.join(pathSave,'%s'%cnty,'data_outputs.xlsx')) as writer:
            dfXY.to_excel(writer,sheet_name='data')
            Yout.to_excel(writer,sheet_name='outputs')
    
    # (4.2) Plot outputs    
    flag = plotOutputsFun(Yout,yearsTrain,YEAR_TEST)    
    if SAVE_OPTION:
        fig.savefig(os.path.join(pathSave,'%s'%cnty,'outputs.png'),dpi=300)
    plt.show()
    
    # Plot only RAstk output for reference, i.e. to compare with single-output GP model    
    flag = plotOutputsFun(Yout.loc[Yout['VarId']==0,:],yearsTrain,YEAR_TEST)    
    if SAVE_OPTION:
        fig.savefig(os.path.join(pathSave,'%s'%cnty,'outputs_ref.png'),dpi=300)
    plt.show()
        
    # # Score
    # # Absolute error
    # Yout['AE'] = abs(Yout['Y'] - Yout['Ymu'])
    # # Absoulte percentage error
    # Yout['APE'] = Yout['AE']/Yout['Ymu']    
