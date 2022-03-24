import glob

import pandas as pd
import numpy as np

import time_series_functions as tsf


def get_models_path(save_path, type_data, models_list, execs):
    models_exec = {}
    for m in models_list:
        paths = glob.glob(f'{save_path}{type_data}-{m}/*')
        if len(paths) == 1:
            paths = paths * execs
        elif len(paths)==execs:
            pass
        else:
            raise Exception(f'execs of {m} is wrong')
        models_exec[m]= paths

    return models_exec

def get_df_models(models_path, e):
    models_prev = {}
    prevs_sizes = []

    for key, item in models_path.items():
        path = item[e]
        models_prev[key] = tsf.open_saved_result(path)['predicted_values']
        prevs_sizes.append( models_prev[key].shape[0])

    prevs_sizes = np.min(prevs_sizes)
    df_prevs = pd.DataFrame()

    for key, item in models_prev.items():
        df_prevs[key] = item[-prevs_sizes:]
    df_prevs = df_prevs.applymap(lambda x: x if x>0 else 0)
    return df_prevs

from sklearn.metrics.pairwise import euclidean_distances

def dinanic_selection(df, real,  test_size, val_size, ds_args):
    rc = ds_args['rc']
    lag_size = ds_args['lag_size']
    k = ds_args['k']

    ts_atu = real[-df.shape[0]:]
    erro_df = df.sub(ts_atu, axis='rows')
    erro_df = erro_df.pow(2)
    val_error = erro_df.iloc[-(test_size+val_size):-test_size]
    val_error.reset_index(drop=True, inplace=True)

    ts_windowed = tsf.create_windowing(lag_size=lag_size,
                                    df=pd.DataFrame({'actual': ts_atu}))

    ts_windowed.drop(columns=['actual'], inplace=True)

    val_windowed = ts_windowed.iloc[-(test_size+val_size):-test_size]
    val_windowed.reset_index(drop=True, inplace=True)

    dists = euclidean_distances(ts_windowed, val_windowed)

    df = df.iloc[lag_size:]
    all_prevs = []
    for d, df_line in zip(dists, df.to_dict('records')):
        d_rc = np.argsort(d)[0:rc]
        error_rc = val_error.loc[list(d_rc)]
        best_models = error_rc.mean().sort_values().iloc[0:k]
        best_models = best_models.index.to_list()
        prevs = [df_line[c] for c in best_models]
        all_prevs.append(np.mean(prevs))

    return all_prevs


def dispach(data, real, save_path, models_list, title, ens_type= 'mean', ds_args={}, execs = 10):
    type_data = data['type_data']
    test_size = data['test_size']
    val_size = data['val_size']

    models_path = get_models_path(save_path, type_data, models_list, execs)

    for e in range(0, execs):
        df = get_df_models(models_path, e)
        if ens_type == 'mean':
            prev_ens = df.mean(axis=1).values

        elif ens_type == 'median':
            prev_ens = df.median(axis=1).values

        elif ens_type == 'ds':
            prev_ens = dinanic_selection(df, real, test_size, val_size, ds_args)
        else:
            raise NotImplementedError(f'{ens_type} not implemented')
    

        ts_atu = real[-len(prev_ens):]
        tsf.make_metrics_avaliation(ts_atu, prev_ens,
                                    test_size, val_size,
                                    tsf.result_options.save_result, ens_type,
                                    title)

     