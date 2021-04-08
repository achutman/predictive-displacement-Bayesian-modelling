# predictive-displacement-Bayesian-modelling
Multi-output Bayesian Gaussian Processes

This software is based on GPy toolbox https://github.com/SheffieldML/GPy.
In particular, please refer to the following links:
https://github.com/SheffieldML/notebook/blob/master/GPy/multiple%20outputs.ipynb
https://nbviewer.jupyter.org/github/SheffieldML/notebook/blob/master/GPy/coregionalized_regression_tutorial.ipynb

The following script models time-series data related to displacement and predictors of displacement, either individually or jointly. The following processed data will be made available in the HDX website. Please refer to updates in the github page for the corresponding links. It is possible to add further relevant time-series datasets.
- UNHCR's refugee and asylum seeker stock
- UN DESA's migrant stock (i.e. super set including both all migrants, forced or not)
- UNHCR's refugee and asylum seeker flow
- UNHCR's returned refugees stock
- World Bank's remittance inflow
- UCDP fatalities

Although the key variable being scored is the UNHCR's refugee and asylum seeker stock data, minor modifications can enable scoring the remaining variables as well.

Scripts:
- gpUniOutRefugeeAsylumScript.py
Models multiple time-series data related to displacement and predictors of displacement individually.
- gpUniOutRefugeeAsylumScriptFuns.py
Relevant function definitions
- gpMultiOutRefugeeAsylumScript.py
Models time-series data related to displacement and predictors of displacement simultaneously.
- gpMultiOutRefugeeAsylumScriptFuns.py
Relevant function definitions
- gpUniMultiOutRefugeeAsylumScoreScript.py
Compares the outputs of uni-output vs. multi-output GP modelling

Outputs:
- Detailed outputs comparing the two modelling techniques for Democratic Republic of Congo.
- Plots comparing the two modeeling techniques for all modelled countries.
- Spreadsheets consolidating outputs from all modelled countries.

Please also refer to the methodoogy.pdf document.
