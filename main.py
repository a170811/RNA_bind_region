#!/usr/bin/env

import os
from functools import reduce

import numpy as np
import pandas as pd
import tensorflow as tf

from utils.dataset import build_data, split
from metrics import *
from networks.base_conv import build_base


accuracy = tf.keras.metrics.BinaryAccuracy(name='acc')
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
specificity = Specificity()
f1 = F1_score()
auc = tf.keras.metrics.AUC()
mcc = MCC()


def ontHot2seq(onehots):# {{{
    inverse_mapping = {
        (1, 0, 0, 0): 'A',
        (0, 1, 0, 0): 'T',
        (0, 0, 1, 0): 'C',
        (0, 0, 0, 1): 'G'
    }
    return [inverse_mapping(tuple(onehot)) for onthot in onehots]
# }}}

def seq2oneHot(seq):# {{{
    mapping = {
        'A': [1., 0., 0., 0.],
        'T': [0., 1., 0., 0.],
        'C': [0., 0., 1., 0.],
        'G': [0., 0., 0., 1.],
    }
    return [mapping[c] for c in seq]
# }}}


def train_and_eval(model_name, seed=0, save=True):# {{{

    path = './data/raw'
    model_name += f'_seed_{seed}'
    model_path = f'./models/{model_name}.h5'

    x, y = build_data(path)
    tr_x, tr_y, va_x, va_y, te_x, te_y = split(x, y, seed=seed)

    tr_pi = np.array([seq2oneHot(seq) for seq in tr_x[:, 0]], dtype=np.float32)
    tr_m = np.array([seq2oneHot(seq) for seq in tr_x[:, 1]], dtype=np.float32)
    va_pi = np.array([seq2oneHot(seq) for seq in va_x[:, 0]], dtype=np.float32)
    va_m = np.array([seq2oneHot(seq) for seq in va_x[:, 1]], dtype=np.float32)
    te_pi = np.array([seq2oneHot(seq) for seq in te_x[:, 0]], dtype=np.float32)
    te_m = np.array([seq2oneHot(seq) for seq in te_x[:, 1]], dtype=np.float32)
    tr_y, va_y, te_y = np.array(tr_y, dtype=np.float32), np.array(va_y, dtype=np.float32), np.array(te_y, dtype=np.float32)

    # deep
    if os.path.exists(model_path) and save:
        print(f'load model: `{model_name}` ...')
        model = tf.keras.models.load_model(model_path, custom_objects={
            'f1': F1_score,
            'specificity': Specificity,
            'mcc': MCC,
        }, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[accuracy, precision, recall, specificity, f1, auc, mcc])
    else:
        model = build_base()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[accuracy, precision, recall, specificity, f1, auc, mcc])
        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)
        model.fit(x=[tr_pi, tr_m], y=tr_y, validation_data=([va_pi, va_m], va_y),\
                  batch_size=512, epochs=1000, callbacks=[callback])
        if save:
            model.save(model_path)

    va_res = model.evaluate(x=[va_pi, va_m], y=va_y, return_dict=True)
    te_res = model.evaluate(x=[te_pi, te_m], y=te_y, return_dict=True)

    va_res = {f'va_{k.split("_")[0]}':v for k, v in va_res.items()}
    te_res = {f'te_{k.split("_")[0]}':v for k, v in te_res.items()}
    return {**va_res, **te_res}
# }}}

if '__main__' == __name__:


    res = train_and_eval('test', save=False)
    print(res)
    exit()

    # v1: base_cnn
    # v2: base_cnn_drop0.4 #2020/07/03
    results = []
    for i in range(30):
        res = train_and_eval('base_cnn_drop0.4', i)
        results.append(res)
    res = pd.DataFrame(results)
    print(res)
    res.to_csv('results_30_times.csv', index=False)

