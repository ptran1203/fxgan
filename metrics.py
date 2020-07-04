import sklearn.metrics as sk_metrics
from keras.utils import to_categorical
from const import CATEGORIES_MAP, INVERT_CATEGORIES_MAP
import numpy as np

def _safe_get(idx):
    try:
        return INVERT_CATEGORIES_MAP[idx]
    except:
        return idx


def markdown_auc(scores, mode_name='VGG16'):
    table = '| Model |'
    size = len(scores)
    for i in range(size):
        table +=  _safe_get(i) + '|'
    table += '\n| -- |'
    for i in range(size):
        table += '-- |'
    table += '\n| VGG16 |'
    for s in scores:
        table += '{} |'.format(round(s, 3))

    print("Average: ", sum(scores) / len(scores))
    return table

def auc_score(y_true, y_pred, verbose=1, plot=0):
    # y_true is not one-hot
    if y_true.shape != y_pred.shape:
        _y_true = to_categorical(y_true, y_pred.shape[-1])
        scores = sk_metrics.roc_auc_score(_y_true, y_pred, average=None)
    else:
        scores = sk_metrics.roc_auc_score(y_true, y_pred, average=None)

    if verbose:
        print('AUC score: \n')
        for i in range(len(scores)):
            print('{}: {}'.format(_safe_get(i), scores[i])) 

    if plot:
        print('[WARN] Function is not implemented')

    return scores

def f1_score(y_true, y_pred, verbose=1):
    _y_pred = np.argmax(y_pred, axis=1)

    scores = sk_metrics.f1_score(y_true, _y_pred, average=None)
    if verbose:
        print('f1 score: \n')
        for i in range(len(scores)):
            print('{}: {}'.format(_safe_get(i), scores[i])) 

    return scores
