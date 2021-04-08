# -*- coding: utf-8 -*-
"""
Created on Thu 18 March 2021 

@author: Achut Manandhar, Data Scientist, Migration and Displacement Initiative, Save the Children International
Contact: William Low, Project Lead, Migration and Displacement Initiative, Save the Children International

The script compares the outputs of gpUniOutRefugeeAsylumScript.py and gpMultiOutRefugeeAsylumScript.py,
i.e. the outputs of uni-output vs. multi-output GP modelling.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import xlrd


# Select countries based on INFORM index
pathInform = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\data\INFORM'
dfInform = pd.read_csv(os.path.join(pathInform,'raw','INFORM2020_TREND_2010_2019_v040_ALL_2 INFORMRiskIndex.csv'))
countries = dfInform.loc[dfInform['2020']>=5,'Iso3'].values
cntySkip = ['PRK','PNG','PSE','SOM']
for cnty in countries:
    if cnty in cntySkip:        
        countries = np.delete(countries,np.where(countries==cnty))

# Path to uni-Out model outputs saved from gpUniOutRefugeeAsylumScript.py
pathSaveUni = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\gaussian process\dataPlots\UNHCR RAstk\timeSepCoregRbfLsFixed3MaxTest2017Nahead1'
# Path to multi-out outputs saved from gpMultiOutRefugeeAsylumScript.py
pathSaveMulti = r'C:\Users\a.manandhar\OneDrive - Save the Children International\Documents\docs\PD phase two model\gaussian process\dataPlots\UNHCR RAstkMstkRretRAflowRemUfat\timeSepCoregRbfLsFixed3MaxTest2017Nahead1'

YEAR_TEST = 2017
Y_RAstk_Uni = pd.DataFrame()
Y_RAstk_Multi = pd.DataFrame()
Y_RAstk_Uni = []
Y_RAstk_Multi = []
for cnty in countries:          
    print(cnty)            
    # Reference = Uni-Output model
    Yout = pd.read_excel(os.path.join(pathSaveUni,cnty,'data_outputs.xlsx'),sheet_name='outputs')    
    # idxTest = ((Yout['Year']>=2017) & (Yout['Year']<=2018))
    idxTest = (Yout['Year']>=2017)
    Y_RAstk_Uni = Y_RAstk_Uni.append(Yout.loc[idxTest,['Year', 'Y', 'Ymu', 'Yvar', 'YmuMinus','YmuPlus']])            
    Y_RAstk_Uni.extend([cnty for i in range(sum(idxTest))])
    # Compare to Multi-Output model
    Yout = pd.read_excel(os.path.join(pathSaveMulti,cnty,'data_outputs.xlsx'),sheet_name='outputs')
    # idxTest = ((Yout['Year']>=2017) & (Yout['Year']<=2018) & (Yout['VarName']=='RAstk'))
    idxTest = ((Yout['Year']>=2017) & (Yout['VarName']=='RAstk'))
    Y_RAstk_Multi = Y_RAstk_Multi.append(Yout.loc[idxTest,['Year', 'Y', 'Ymu', 'Yvar', 'YmuMinus','YmuPlus']])            
    Y_RAstk_Multi.extend([cnty for i in range(sum(idxTest))])

# AE = Absolute Error
# APE = Absolute Percentage Error
# CI = 95% Confidence Interval
Y_RAstk_Uni['AE'] = np.round_(np.abs(Y_RAstk_Uni['Ymu']-Y_RAstk_Uni['Y']))
Y_RAstk_Uni['APE'] = np.round_(100*np.abs(Y_RAstk_Uni['Ymu']-Y_RAstk_Uni['Y'])/Y_RAstk_Uni['Y'])
Y_RAstk_Uni['CI'] = np.round_(Y_RAstk_Uni['YmuPlus']-Y_RAstk_Uni['YmuMinus'])
Y_RAstk_Uni['Country'] = np.array(Y_RAstk_Uni)
#
Y_RAstk_Multi['AE'] = np.round_(np.abs(Y_RAstk_Multi['Ymu']-Y_RAstk_Multi['Y']))
Y_RAstk_Multi['APE'] = np.round_(100*np.abs(Y_RAstk_Multi['Ymu']-Y_RAstk_Multi['Y'])/Y_RAstk_Multi['Y'])
Y_RAstk_Multi['CI'] = np.round_(Y_RAstk_Multi['YmuPlus']-Y_RAstk_Multi['YmuMinus'])
Y_RAstk_Multi['Country'] = np.array(Y_RAstk_Multi)

Y_RAstk_Uni = Y_RAstk_Uni.reset_index().drop('index',axis=1)
Y_RAstk_Multi = Y_RAstk_Multi.reset_index().drop('index',axis=1)

# Save score on test data
# Y_RAstk_Uni = Y_RAstk_Uni.set_index('Country')
# Y_RAstk_Uni.to_csv(os.path.join(pathSave,'score.csv'))
# Y_RAstk_Multi = Y_RAstk_Multi.set_index('Country')
# Y_RAstk_Multi.to_csv(os.path.join(pathSaveMulti,'score.csv'))

# Aggregate scores
print(Y_RAstk_Uni.loc[:,['Year','AE','APE','CI']].groupby('Year').median())
print(Y_RAstk_Multi.loc[:,['Year','AE','APE','CI']].groupby('Year').median())

# Plot score comparison
SAVE_OPTION = 0
for year in [2017,2018]:
    fig, ax = plt.subplots()
    # Compare distribution of AE Uni-out vs. Multi-out
    ax.hist(np.log10(Y_RAstk_Uni.loc[Y_RAstk_Uni['Year']==year,'AE'].values),color='C1',alpha=.25)
    ax.hist(np.log10(Y_RAstk_Multi.loc[Y_RAstk_Multi['Year']==year,'AE'].values),color='C4',alpha=.25)
    # Compare distribution of APE Uni-out vs. Multi-out
    # ax.hist(Y_RAstk_Uni.loc[Y_RAstk_Uni['Year']==year,'APE'].values,color='C1',alpha=.25)
    # ax.hist(Y_RAstk_Multi.loc[Y_RAstk_Multi['Year']==year,'APE'].values,color='C4',alpha=.25)
    ax.set_title('Bayesian Modelling Distribution of Error for %d Estimates'%year)
    ax.set_xlabel('log10[Absolute Error]')    
    ax.set_ylabel('Number of Countries')    
    fig.legend(['Single-Output','Multi-Output'],loc='center')
    fig.tight_layout()
    if SAVE_OPTION:
        fig.savefig(os.path.join(pathSaveMulti,'score_comparison_%d.png'%year),dpi=300)
    plt.show()
    plt.show()

# Compare AE Uni-out vs. Multi-out for APE<50%
APEthresh = 50
idxKeep = (Y_RAstk_Uni['APE']<APEthresh)
yUni = Y_RAstk_Uni.loc[idxKeep,'APE'].values
yMulti = Y_RAstk_Multi.loc[idxKeep,'APE'].values

fig,ax=plt.subplots(figsize=(6,5))
ax.plot(yUni,yMulti,'o')
ax.plot([0,APEthresh],[0,APEthresh],':k')
ax.set_xlim([-1,APEthresh])
ax.set_ylim([-1,APEthresh])
ax.set_xlabel(r'$\%$ Error (Uni-Out)')
ax.set_ylabel('Percentage Error (Multi-Out)')
ax.set_title(r'$\%$ Error Comparison across %d Countries'%(len(Y_RAstk_Uni.loc[idxKeep,'Country'].unique())))
