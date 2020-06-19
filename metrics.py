from sklearn.metrics import roc_auc_score
from keras.utils import to_categorical
from const import CATEGORIES_MAP, INVERT_CATEGORIES_MAP

def auc_score(y_true, y_pred, verbose=1, plot=0):
    # y_true is not one-hot
    if y_true.shape != y_pred.shape:
        _y_true = to_categorical(y_true, y_pred.shape[-1])
        scores = roc_auc_score(_y_true, y_pred, average=None)
    else:
        scores = roc_auc_score(y_true, y_pred, average=None)
    
    if verbose:
        print('AUC score: \n')
        print(scores)
        for i in range(len(scores)):
            print(scores[i])
            print('{}: {}'.format(INVERT_CATEGORIES_MAP[i], scores[i])) 

    if plot:
        print('[WARN] Function is not implemented')

    return scores