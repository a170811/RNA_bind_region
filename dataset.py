#!/usr/bin/env python3

import pickle
from pathlib import Path

from build_data.utils import build_data, split


data_path = './data/raw/'
save_path = './data/0602/'

x, y = build_data(data_path)

Path(save_path).mkdir(parents=True, exist_ok=False)
tr_x, tr_y, va_x, va_y, te_x, te_y = split(x, y, ratio=[0.8, 0.1, 0.1])

pickle.dump((tr_x, tr_y), open(f'{save_path}/tr.pkl', 'wb'))
pickle.dump((va_x, va_y), open(f'{save_path}/va.pkl', 'wb'))
pickle.dump((te_x, te_y), open(f'{save_path}/te.pkl', 'wb'))

print(f'All done, save to {save_path}')
