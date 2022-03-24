import os
import json
import time

import pandas as pd
import numpy as np
from sklearn.model_selection import ParameterGrid
from sklearn.base import clone
from tqdm import tqdm

import time_series_functions as tsf
import fit_predict_models as fpm


def utc_hour_to_int(x):
    return int(x.split(' ')[0])

def load_data_solar_hours(path, min_max, use_log, save_cv):
    df_total = pd.read_csv(path, sep=';',decimal=',', encoding='utf8')
    df_total['RADIACAO GLOBAL (Kj/m)'] = df_total['RADIACAO GLOBAL (Kj/m)'].fillna(0)
    df_total['Data'] = pd.to_datetime(df_total['Data'] +' '+ df_total['Hora UTC'], format='%d/%m/%Y %H%M UTC')
    
    min_hour =  utc_hour_to_int(min_max[0])
    max_hour = utc_hour_to_int(min_max[1])
    cond = df_total['Hora UTC'].apply(lambda x: utc_hour_to_int(x)>=min_hour and utc_hour_to_int(x)<=max_hour)
    df_total = df_total[cond][['Data', 'RADIACAO GLOBAL (Kj/m)']]
    
    df_total.rename(columns = {'RADIACAO GLOBAL (Kj/m)': 'actual'}, inplace=True)
    df_total.set_index('Data', inplace=True)

    if use_log:
        df_total['actual'] = np.log(df_total['actual']+1)

    if save_cv:
        df_total.to_csv(path.replace('.csv', '_solar.csv')) 

    return df_total


def do_grid_search(type_data, real, test_size, val_size, parameters, model, horizon,
                   recurvise, use_exegen_future, model_execs):


    best_model = None
    metric = 'RMSE'
    best_result = {'time_window':0,metric:None}
    result_type = tsf.result_options.val_result

    list_params=list(ParameterGrid(parameters))
    
    for params in tqdm(list_params,desc='GridSearch'):
        
        result = None
        params_actual = params.copy()
        del params_actual['time_window']

        forecaster = clone(model).set_params(** params_actual)
                    
        result_atual = []
        for t in range(0,model_execs):
            result_atual.append(fpm.single_model('mlp',type_data, params['time_window'],real,
                                      forecaster,test_size,val_size,
                                      result_type,True, horizon, recurvise, use_exegen_future)[metric])

        result = np.mean(np.array(result_atual))

        if best_result[metric] == None:
            best_model = forecaster
            best_result[metric] = result
            best_result['time_window'] = params['time_window']
        else:

            if best_result[metric] > result:
                best_model = forecaster
                best_result[metric] = result
                best_result['time_window'] = params['time_window']

    result_model = {'best_result': best_result, 'model': best_model}
    return result_model


def train_sklearn(model_execs, data_title, parameters, model):
    
    config_path = './'
    save_path = './solar_rad/'
    with open(f'{config_path}models_configuration_60_20_20.json') as f:
        data = json.load(f)

    recurvise = False
    use_exegen_future = False
    use_log = False
    
    for i in data:

        if i['activate']==1:

            print(i['name'])
            print(i['path_data'])
            test_size=i['test_size']
            val_size=i['val_size']
            type_data = i['type_data']
            horizon = i['horzion']
            min_max = i['hour_min_max']

            real = load_data_solar_hours(i['path_data'], min_max, use_log, True)

            gs_result = do_grid_search(type_data=type_data,
                                       real=real,test_size=test_size,
                                       val_size=val_size,
                                       parameters=parameters,
                                       model=model,
                                       horizon=horizon, 
                                       recurvise=recurvise,
                                       use_exegen_future=use_exegen_future,
                                      model_execs=model_execs)

            print(gs_result)
            save_path_actual = save_path+str(type_data)+'-'+data_title+'/'
            os.mkdir(save_path_actual)

            title_temp = str(type_data)+ '-' + data_title
            for _ in range(0, model_execs):
                fpm.single_model(save_path_actual+title_temp,type_data,gs_result['best_result']['time_window'],
                                 real, 
                                 gs_result['model'],test_size,val_size,tsf.result_options.save_result,True, horizon,
                                recurvise, use_exegen_future)
                time.sleep(1)
