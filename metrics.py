import sklearn.metrics as sk_metrics
from keras.utils import to_categorical
from const import CATEGORIES_MAP, INVERT_CATEGORIES_MAP
import numpy as np

def _safe_get(idx):
    try:
        return INVERT_CATEGORIES_MAP[idx]
    except:
        return idx


def draw_md_table(scores):
    """
    sample input:
    {
        'VGG16': [0.739 ,0.786 ,0.721 ,0.755 ,0.776 ,0.774 ,0.683 ,0.884],
        'standard aug': [0.731 ,0.779 ,0.721 ,0.771 ,0.774 ,0.755 ,0.690 ,0.865],
        'GAN v1': [0.744 ,0.774 ,0.736 ,0.772 ,0.775 ,0.757 ,0.695 ,0.878],
    }
    """
    table = '|  |'
    for name in scores.keys():
        table += ' {} |'.format(name)

    table += '\n|'
    for i in range(len(scores) + 1):
        table += '--|'

    table += '\n'
    head = scores[list(scores.keys())[0]]
    len_head = len(head)
    avgs = [sum(v)/len_head for v in scores.values()]
    for i in range(len_head):
        # use i + 1 because we don't care No Finding case 
        table += '| ' + _safe_get(i) + ' |'
        # find the best score value
        best = 0
        row = ''
        for name in scores.keys():
            point = round(scores[name][i], 3)
            if point > best:
                best = point
            row += ' {} |'.format(point)
        row = row.replace(str(best), '**{}**'.format(best))
        table += row + '\n'
    row = '| **Average** |'
    best = 0
    for avg in avgs:
        avg = round(avg, 3)
        if avg > best:
            best = avg
        row += ' {} |'.format(avg)
    row = row.replace(str(best), '**{}**'.format(best))
    table += row + '\n'
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
