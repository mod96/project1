from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import lfilter, savgol_filter, butter, filtfilt
import statsmodels.api as sm

from TaPR_pkg import etapr

from settings import MODEL_SAVE_FOLDER

from .RangeCheck import range_check


GLOBAL_FILTER = None


def test_and_get_list(model, dataset_list, raw_dataset_list, args, **kwargs):
    loss_list = []
    time_len = args.sequence_length
    for dataset, raw_dataset in zip(dataset_list, raw_dataset_list):
        res = np.zeros(time_len, dtype='float32')  # padding
        predicted = model.predict(dataset)
        loss = np.mean(np.square(predicted - raw_dataset[time_len:]), axis=-1).astype('float32')
        res = np.concatenate((res, loss), axis=0)
        assert len(res) == len(raw_dataset), f"padding seems wrong, {len(res)} != {len(raw_dataset)}"
        loss_list.append(res)

    res = loss_list[0]
    for loss in loss_list[1:]:
        res = np.concatenate((res, loss), axis=0)

    if args.anomaly_smoothing != "NO":
        res = smoothing(res, args, **kwargs)

    if args.range_check_window:
        res = range_check(res, args.range_check_window)

    return res


def find_th(args, attack, valid_result, start=0.0002, end=0.01, division=2000, view_plt=True):
    res = []
    for inc in tqdm(range(division), desc="Finding Threshold"):
        th = start + inc * (end - start) / division
        final = put_labels(valid_result, th)
        TaPR = etapr.evaluate_haicon(anomalies=attack, predictions=final)
        res.append((TaPR['f1'] - abs(TaPR['TaP'] - TaPR['TaR'])/3, TaPR['TaP'], TaPR['TaR'], th))

    _, _, _, threshold = max(res)
    final = put_labels(valid_result, threshold)
    TaPR = etapr.evaluate_haicon(anomalies=attack, predictions=final)
    print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
    print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")
    print("Threshold : ", threshold)

    if view_plt:
        check_graph(args, valid_result, final, piece=5, THRESHOLD=threshold)

    return threshold, [TaPR['f1'], TaPR['TaP'], TaPR['TaR']]


def put_labels(distance, threshold):
    xs = np.zeros_like(distance)
    xs[distance > threshold] = 1
    return xs


def check_graph(args, xs, att, piece=2, THRESHOLD=None):
    l = xs.shape[0]
    chunk = l // piece
    fig, axs = plt.subplots(piece, figsize=(20, 4 * piece))
    for i in range(piece):
        L = i * chunk
        R = min(L + chunk, l)
        xticks = range(L, R)
        axs[i].plot(xticks, xs[L:R])
        if len(xs[L:R]) > 0:
            peak = max(xs[L:R])
            axs[i].plot(xticks, att[L:R] * peak * 0.3)
        if THRESHOLD!=None:
            axs[i].axhline(y=THRESHOLD, color='r')
    plt.savefig(f'{MODEL_SAVE_FOLDER}/{args.model_name}/check_graph.png')
    plt.show()


def smoothing(res, args, attack=None):  # 1d
    filter_type = args.anomaly_smoothing
    if filter_type == "rolling":
        return filter0_factory(args.smoothing_parameter)(res)

    elif filter_type == "lfilter":
        return filter1_factory(args.smoothing_parameter)(res)

    elif filter_type == "savgol":
        return filter2_factory(args.smoothing_parameter)(res)

    elif filter_type == "lowess":
        return filter3_factory(args.smoothing_parameter)(res)

    elif filter_type == "search_all":
        global GLOBAL_FILTER
        if GLOBAL_FILTER == None:
            GLOBAL_FILTER = noise_filtering_test_all(res, attack, args)

        return GLOBAL_FILTER(res)

    return res


def filter0_factory(window_size):
    def filter(val_res):
        anomaly_score_series = pd.Series(val_res)
        return anomaly_score_series.rolling(window=int(window_size), min_periods=1).mean().values
    return filter


def filter1_factory(n):  # Filter data along one-dimension with an IIR or FIR filter.
    n = int(n)
    def filter(val_res):
        b = [1.0 / n] * n
        a = 1
        return lfilter(b,a,val_res)
    return filter


def filter2_factory(window_size):  # Apply a Savitzky-Golay filter to an array.
    def filter(val_res):
        return savgol_filter(val_res, int(window_size), 2)
    return filter


def filter3_factory(fractional):  # LOWESS
    def filter(val_res):
        x = list(range(len(val_res)))
        y_lowess = sm.nonparametric.lowess(val_res, x, frac=fractional)
        return y_lowess[:, 1]
    return filter


def filter4_factory(window_size):
    def filter(val_res):
        anomaly_score_series = pd.Series(val_res)
        return anomaly_score_series.rolling(window=int(window_size), min_periods=1).median().values
    return filter


def filter5_factory(N, Wn):
    b, a = butter(N, Wn)
    def filter(val_res):
        return filtfilt(b, a, val_res)
    return filter


def noise_filtering_test_all(validation_result, attack, args):
    def noise_filtering_test(parameters, filters):
        local_res = []
        for p, f in zip(parameters, filters):
            print(f"parameter : {p}")
            try:
                var_res = f(validation_result)
                _, temp = find_th(args, attack, var_res, view_plt=False)
                local_res.append(temp + [p, f])
            except:
                print("this failed")
                pass
        return local_res

    res = []
    parameters = [5, 10, 15, 20, 40, 50, 60, 70, 80, 100]
    filters = [filter0_factory(ws) for ws in parameters]
    res += noise_filtering_test(parameters, filters)

    parameters = [2, 4, 8, 16, 32, 64, 128]
    filters = [filter1_factory(n) for n in parameters]
    res += noise_filtering_test(parameters, filters)

    parameters = [5, 11, 15, 21, 41, 51, 61, 71, 81, 101]
    filters = [filter2_factory(ws) for ws in parameters]
    res += noise_filtering_test(parameters, filters)

    parameters = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3]
    filters = [filter3_factory(f) for f in parameters]
    res += noise_filtering_test(parameters, filters)

    parameters = [5, 10, 15, 20, 40, 50, 60, 70, 80, 100]
    filters = [filter4_factory(ws) for ws in parameters]
    res += noise_filtering_test(parameters, filters)

    parameters = [(1, 0.1), (1, 0.2), (1, 0.05), (2, 0.1), (2, 0.2), (2, 0.05)]
    filters = [filter5_factory(N, Wn) for N, Wn in parameters]
    res += noise_filtering_test(parameters, filters)

    print(res)

    final = max(res, key=lambda li: (li[0], li[1], li[2]))

    print(final)

    return final[-1]
