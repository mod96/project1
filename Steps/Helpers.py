from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from TaPR_pkg import etapr

from settings import MODEL_SAVE_FOLDER


def test_and_get_list(model, dataset_list, raw_dataset_list, timelen):
    loss_list = []

    for dataset, raw_dataset in zip(dataset_list, raw_dataset_list):
        res = np.zeros(timelen, dtype='float32')  # padding
        predicted = model.predict(dataset)
        loss = np.mean(np.abs(predicted - raw_dataset[timelen:]), axis=-1).astype('float32')
        res = np.concatenate((res, loss), axis=0)
        assert len(res) == len(raw_dataset), f"padding seems wrong, {len(res)} != {len(raw_dataset)}"
        loss_list.append(res)

    res = loss_list[0]
    for loss in loss_list[1:]:
        res = np.concatenate((res, loss), axis=0)
    return res


def find_th(args, attack, valid_result, start=0.01, end=0.1, division=1000):
    res = []
    for inc in tqdm(range(division), desc="Finding Threshold"):
        th = start + inc * (end - start) / division
        final = put_labels(valid_result, th)
        TaPR = etapr.evaluate_haicon(anomalies=attack, predictions=final)
        res.append((TaPR['f1'], TaPR['TaP'], TaPR['TaR'], th))

    _, _, _, threshold = max(res)
    final = put_labels(valid_result, threshold)
    TaPR = etapr.evaluate_haicon(anomalies=attack, predictions=final)
    print(f"F1: {TaPR['f1']:.3f} (TaP: {TaPR['TaP']:.3f}, TaR: {TaPR['TaR']:.3f})")
    print(f"# of detected anomalies: {len(TaPR['Detected_Anomalies'])}")
    print(f"Detected anomalies: {TaPR['Detected_Anomalies']}")
    print("Threshold : ", threshold)
    check_graph(args, valid_result, final, piece=5, THRESHOLD=threshold)
    return threshold


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
