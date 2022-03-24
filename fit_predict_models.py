import time_series_functions as tsf
import pandas as pd
import numpy as np
from sklearn import preprocessing

def get_windowing(ts_normalized , time_window, horizon , prefix=''):
    ts_windowed = tsf.create_windowing(lag_size=(time_window + (horizon-1)), 
                                        df=ts_normalized)

    columns_lag = [f'lag_{l}{prefix}'for l in reversed(range(1,time_window+1))]
    columns_horizon = [f'hor_{l}{prefix}'for l in range(1,horizon)] + ['actual']
    ts_windowed.columns= columns_lag + columns_horizon

    ts_windowed = ts_windowed[columns_lag+['actual']]
    return ts_windowed

def single_model(title, type_data, time_window, time_series, model, test_size,
                 val_size, return_option, normalize, horizon=1, recursive=False, use_exo_future=True):
    train_size = len(time_series) - test_size

    is_exogen = False
    
    if time_series.shape[1]> 1:
        is_exogen = True
        exogens = time_series.drop(columns = ['actual', 'Data'], errors='ignore')
    horizon_to_use = horizon
    if recursive:
        if is_exogen:
            raise NotImplementedError('RECUSIVE IS NOT SUPPORTED WITH EXOGENS')
        horizon=1

    # normalize
    if normalize:
        min_max_scaler = preprocessing.MinMaxScaler()
        min_max_scaler.fit(time_series['actual'].values[0:train_size].reshape(-1, 1))
        ts_normalized = min_max_scaler.transform(time_series['actual'].values.reshape(-1, 1))
        ts_normalized = pd.DataFrame({'actual': ts_normalized.flatten()})

        if is_exogen:
            min_max_scaler_x = preprocessing.MinMaxScaler()
            min_max_scaler_x.fit(exogens.values[0:train_size])
            exogens_norm = min_max_scaler_x.transform(exogens)
            exogens_norm = pd.DataFrame(exogens_norm, columns=exogens.columns)

    else:
        ts_normalized = time_series
        if is_exogen:
            exogens_norm = exogens
    # ________________

    ts_windowed = get_windowing(ts_normalized , time_window, horizon)
    if is_exogen:
        exgen_windowed = pd.DataFrame()
        for c in exogens.columns:
            if use_exo_future:
                df_exogen = get_windowing(exogens_norm[c], time_window, horizon, f'_{c}')[['actual']]
                df_exogen.rename(columns={'actual':c}, inplace=True)
            else:
                df_exogen = get_windowing(exogens_norm[c], time_window, horizon, f'_{c}').drop(columns=['actual'])

            exgen_windowed = pd.concat([exgen_windowed, df_exogen], axis=1)
        
        ts_windowed = pd.concat([exgen_windowed, ts_windowed], axis=1)

    reg = tsf.fit_sklearn_model(ts_windowed, model, test_size, val_size)
    
    if recursive and (horizon_to_use>1):
        ts_windowed_test = get_windowing(ts_normalized , time_window, horizon_to_use)
        ts_windowed_test = ts_windowed_test.iloc[-(test_size+val_size+10):]
        pred = tsf.predict_sklearn_model_recursive(ts_windowed_test, reg, horizon_to_use)
    else:
        pred = tsf.predict_sklearn_model(ts_windowed, reg)

    if(normalize):
        pred = min_max_scaler.inverse_transform(pred.reshape(-1, 1)).flatten()

    ts_atu = time_series['actual']
    ts_atu = ts_atu[-len(pred):]

    try:
        df_prevs = model.prevs_df
    except:
        df_prevs = None
    
    results = tsf.make_metrics_avaliation(ts_atu, pred,
                                          test_size, val_size,
                                          return_option, model.get_params(deep=True),
                                          title + '(tw' + str(time_window) + ')', df_prevs)
    return results