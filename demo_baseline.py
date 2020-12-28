import os
import numpy as np
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Incremental learning demonstration')
parser.add_argument('--stats', '-s', action='store_true', help='only read stats dataframe')
args = parser.parse_args()

fname = 'stats.pkl'

if not args.stats:
    # https://github.com/pytorch/pytorch/issues/37377
    os.environ['MKL_THREADING_LAYER'] = 'GNU'

    class_start = 50
    class_end = 100
    step = 10
    lr = 0.2
    epochs = 70
    tde = '--test_data_exposure'

    parameters_lst = []
    for c in range(class_start, class_end+1, step):
        if c == class_start:
            parameters_lst.append((tde, lr, epochs, c, '', ''))
        elif c == class_start + step:
            parameters_lst.append(('', lr, epochs, c, c - step, ''))
        else:
            parameters_lst.append(('', lr, epochs, c, c - step, c - 2 * step))

    for parameters in parameters_lst:
        tde, lr, epochs, nc, nco, ncoo = parameters
        os.system(f'python baseline.py {tde} --lr {lr} --epochs {epochs} --num_classes {nc} --num_classes_old {nco} --num_classes_old_old {ncoo} --stats_fname {fname}')

print(f"Statistics are saved at ./stats/{fname}\n")
df = pd.read_pickle(f'stats/{fname}')
pd.options.display.max_columns = 2000
pd.options.display.max_colwidth = 2000
print(df)