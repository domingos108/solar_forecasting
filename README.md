# solar_forecasting
the source code of the paper named as: Solar Irradiance forecasting using dynamic ensemble selection


# Usage
General information: 
- There are a jupyter notebook to execute each single model (TODO: ARIMA execution);
- After execute all single model, is possible use the notebook called by heterogen_ensemble.ipynb, to create the Dinamic Selection experiments;

Data Files information:
- The file models_configuration_60_20_20.json is responsible to configure the time series paths, test and validation splits,  series names and time series codes;
- All time series data are inside of ./data/;
- All models execution will be persisted in ./solar_rad/;
- The ARIMA model are persisted in ./arima_models/;

Python files description:
- fit_predict_models.py: resposible for training and predict scripts for Sklearn based models;
- heterogen_ensemble.py: resposible for create dinamic selection approaches;
- time_series_functions.py: scripts for persist and load results, and compute performance metrics;
- utils.py: utilities for load the solar data and perform the grid-search of the single models;
