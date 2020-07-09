#!/usr/bin/env


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-mode', '--mode', help='blabla', type=str)
parser.add_argument('-tr', '--train', help='blabla', type=str)
parser.add_argument('-va', '--validation', help='blabla', type=str)
parser.add_argument('-te', '--testing', help='blabla', type=str)
parser.add_argument('-model', '--model', help='blabla', type=str)
args = parser.parse_args()


import numpy as np
import pandas as pd
import tensorflow as tf

from metrics import *
from networks.transformer import build_transformer_base
from networks.transformer import MultiHeadSelfAttention, TransformerBlock, TokenAndPositionEmbedding


accuracy = tf.keras.metrics.BinaryAccuracy(name='acc')
precision = tf.keras.metrics.Precision()
recall = tf.keras.metrics.Recall()
specificity = Specificity()
f1 = F1_score()
auc = tf.keras.metrics.AUC()
mcc = MCC()


def seq2label(seq):# {{{
    mapping = {
        'A': 0,
        'T': 1,
        'C': 2,
        'G': 3,
    }
    return [mapping[c] for c in seq]
# }}}

def learning_rate_scheduler(epoch):# {{{
    if epoch < 10:
        return 0.001
    else:
        return 0.001 * 0.1 ** (epoch/10)
# }}}

def load_csv(path):# {{{
    data = pd.read_csv(path)
    data = data.to_numpy()
    pi = np.array([seq2label(seq) for seq in data[:, 0]], dtype=np.float32)
    m = np.array([seq2label(seq) for seq in data[:, 1]], dtype=np.float32)
    y = np.array(data[:, 2], dtype=np.float32)
    return pi, m, y
# }}}


if '__main__' == __name__:

    data_path = './inputdata/'

    if 'all' == args.mode:
        assert args.train, 'training data missing'
        assert args.validation, 'validation data missing'
        assert args.testing, 'testing data missing'

        output_path = './all/'

        tr_pi, tr_m, tr_y = load_csv(f'{data_path}/{args.train}')
        va_pi, va_m, va_y = load_csv(f'{data_path}/{args.validation}')
        te_pi, te_m, te_y = load_csv(f'{data_path}/{args.testing}')

        model = build_transformer_base()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[accuracy, precision, recall, specificity, f1, auc, mcc])
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)
        scheduler = tf.keras.callbacks.LearningRateScheduler(learning_rate_scheduler)
        model.fit(x=[tr_pi, tr_m], y=tr_y, validation_data=([va_pi, va_m], va_y),\
                  batch_size=512, epochs=1000, callbacks=[early_stop, scheduler])
        model.save(f'{output_path}/model.h5')

        va_res = model.evaluate(x=[va_pi, va_m], y=va_y, return_dict=True)
        te_res = model.evaluate(x=[te_pi, te_m], y=te_y, return_dict=True)
        result = pd.DataFrame({k: [va_res[k], te_res[k]] for k in va_res.keys()}, index=['va', 'te'])

    elif 'test' == args.mode:
        assert args.model, 'model name missing'
        assert args.testing, 'testing data missing'

        model_path = f'./test/{args.model}'
        output_path = './test/'

        te_pi, te_m, te_y = load_csv(f'{data_path}/{args.testing}')

        print(f'load model: `{args.model}` ...')
        model = tf.keras.models.load_model(model_path, custom_objects={
            'f1': F1_score,
            'specificity': Specificity,
            'mcc': MCC,
            'MultiHeadSelfAttention': MultiHeadSelfAttention,
            'TransformerBlock': TransformerBlock,
            'TokenAndPositionEmbedding': TokenAndPositionEmbedding,
        }, compile=False)
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[accuracy, precision, recall, specificity, f1, auc, mcc])

        te_res = model.evaluate(x=[te_pi, te_m], y=te_y, return_dict=True)
        result = pd.DataFrame(te_res, index=['te'])
    else:
        print('mode missing')
        exit()

    print(result)
    result.to_csv(f'{output_path}/results.csv', index=False)
