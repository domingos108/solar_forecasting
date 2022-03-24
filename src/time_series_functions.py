import pickle as pkl
import datetime
from collections import deque

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class result_options:
    test_result = 0
    val_result = 1
    train_result = 2
    save_result = 3

def create_windowing(df, lag_size):
    final_df = None
    for i in range(0, (lag_size + 1)):
        serie = df.shift(i)
        if (i == 0):
            serie.columns = ['actual']
        else:
            serie.columns = [str('lag' + str(i))]
        final_df = pd.concat([serie, final_df], axis=1)

    return final_df.dropna()


def mean_square_error(y_true, y_pred):
    y_true = np.asmatrix(y_true).reshape(-1)
    y_pred = np.asmatrix(y_pred).reshape(-1)

    return np.square(np.subtract(y_true, y_pred)).mean()

def root_mean_square_error(y_true, y_pred):

    return mean_square_error(y_true, y_pred)**0.5


def mean_absolute_percentage_error(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    posi_with_zeros = np.where(y_true == 0)[0]

    y_true = [n for k, n in enumerate(y_true) if k not in posi_with_zeros]
    y_pred = [n for k, n in enumerate(y_pred) if k not in posi_with_zeros]
    
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def mean_absolute_error(y_true, y_pred):
    
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    return np.mean(np.abs(y_true - y_pred))


def average_relative_variance(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.subtract(y_true, y_pred)).sum()
    error_inf = np.square(np.subtract(y_pred, mean)).sum()

    return error_sup / error_inf


def index_agreement(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)
    mean = np.mean(y_true)

    error_sup = np.square(np.abs(np.subtract(y_true, y_pred))).sum()

    error_inf = np.abs(np.subtract(y_pred, mean)) + np.abs(np.subtract(y_true, mean))
    error_inf = np.square(error_inf).sum()

    return 1 - (error_sup / error_inf)


def make_metrics_avaliation(y_true, y_pred, test_size,
                            val_size,return_type,model_params,
                            title, prevs_df=None):
    data_size = len(y_true)
    train_size = data_size - (val_size + test_size)

    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    y_true_test = y_true[(data_size - test_size):data_size]
    y_pred_test = y_pred[(data_size - test_size):data_size]

    val_result = None

    if val_size>0:
        y_true_val = y_true[(train_size):(data_size - test_size)]
        y_pred_val = y_pred[(train_size):(data_size - test_size)]
        val_result = gerenerate_metric_results(y_true_val, y_pred_val)

    y_true_train = y_true[:train_size]
    y_pred_train = y_pred[:train_size]

    geral_dict = {
        'test_metrics': gerenerate_metric_results(y_true_test, y_pred_test),
        'val_metrics': val_result,
        'train_metrics': gerenerate_metric_results(y_true_train, y_pred_train),
        'real_values': y_true,
        'predicted_values': y_pred,
        'pool_prevs': prevs_df,
        'params': model_params
    }

    if return_type == 0:
        return geral_dict['test_metrics']
    elif return_type == 1:
        return geral_dict['val_metrics']
    elif return_type == 2:
        return geral_dict['train_metrics']
    elif return_type == 3:
        return save_result(geral_dict, title)


def direction_acerted(real, prev):
    if real < 0 and prev < 0:
        return 1
    elif real > 0 and prev > 0:
        return 1
    elif real == 0 and prev == 0:
        return 1

    return 0


def metric_direction_acerted(real, prevs):
    size_total = len(real)
    directions = [direction_acerted(real[s], prevs[s]) for s in range(0, size_total)]
    porcent_acerted = np.sum(directions) / size_total

    return porcent_acerted

def gerenerate_metric_results(y_true, y_pred):

    return {'MSE': mean_square_error(y_true, y_pred),
            'RMSE':root_mean_square_error(y_true, y_pred),
            'MAPE': mean_absolute_percentage_error(y_true, y_pred),
            'MAE': mean_absolute_error(y_true, y_pred),
            'ARV': average_relative_variance(y_true, y_pred),
            'IA': index_agreement(y_true, y_pred),
            }

import uuid
def save_result(dict_result, title):

    currentDT = datetime.datetime.now()
    title = title+"-"+str(uuid.uuid4())+".pkl"
    
    with open(title, 'wb') as handle:
        pkl.dump(dict_result, handle)

    #print("exported to pkl")
    return title

def open_saved_result(file_name):
    
    with open(file_name, 'rb') as handle:
        b = pkl.load(handle)
    return b


def fit_sklearn_model(ts, model, test_size, val_size):
    train_size = ts.shape[0] - test_size - val_size
    y_train = ts['actual'][0:train_size]
    x_train = ts.drop(columns=['actual'], axis=1)[0:train_size]
    
    return model.fit(x_train.values, y_train.values)


def predict_sklearn_model(ts, model):

    x = ts.drop(columns=['actual'], axis=1)

    return model.predict(x.values)

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
