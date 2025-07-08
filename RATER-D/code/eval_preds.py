import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from netcal.metrics import ECE

from pprint import pprint
import sys
import os

df = pd.read_csv(sys.argv[1].replace('/', '_') + '.csv')

model_name = os.path.basename(sys.argv[1].replace('/', '_'))

y_true = df['target']
preds = np.where(df['pred'] >= 5, 1, 0)
probs = (df['pred'] - 1) / 6

tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
auc = np.round(roc_auc_score(y_true, probs), 3)
macro_f1 = np.round(f1_score(y_true, preds, average = 'macro'), 3)

n_bins = 7
ece = np.round(ECE(bins = n_bins).measure(np.array(probs), np.array(y_true)), 3)

metrics = {
    'model': model_name.replace('_', '/'),
    'tn': tn,
    'fp': fp,
    'fn': fn,
    'tp': tp,
    'auc': auc,
    'macro_f1': macro_f1,
    'ece': ece
}

pprint(metrics, sort_dicts = False)