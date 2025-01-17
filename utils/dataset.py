import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def build_data(path):

    pos = pd.read_csv(f'{path}/pos.csv', index_col=False)[['piRNA_seq', 'mRNA_seq']]
    neg = pd.read_csv(f'{path}/neg.csv', index_col=False)[['piRNA_seq', 'mRNA_seq']]
    pos['ans'] = 1
    neg['ans'] = 0

    pos, neg = pos.to_numpy(), neg.to_numpy()
    data = np.concatenate([pos, neg], axis=0) # (389891, 3)

    return data[:, :2], data[:, 2]


def split(x, y, ratio=[0.8, 0.1, 0.1], seed=0):
    # ratio: tr, va, te

    assert 1 == np.sum(ratio), f'sum of train, valid, test must be 1'
    tr_x, va_x, tr_y, va_y = train_test_split(x, y, train_size=ratio[0], random_state=seed)
    train_size = ratio[1] / sum(ratio[1:])
    va_x, te_x, va_y, te_y = train_test_split(va_x, va_y, train_size=train_size, random_state=seed)
    return tr_x, tr_y, va_x, va_y, te_x, te_y


if '__main__' == __name__:

    path = '../data/raw/'
    x, y = build_data(path)
