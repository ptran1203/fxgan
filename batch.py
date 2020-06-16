import numpy as np


def pure_batch_norm(X, gamma, beta, eps = 1e-5):
    if len(X.shape) not in (2, 4):
        raise ValueError('only supports dense or 2dconv')

    # dense
    if len(X.shape) == 2:
        # mini-batch mean
        mean = np.mean(X, axis=0)
        # mini-batch variance
        variance = np.mean((X - mean) ** 2, axis=0)
        # normalize
        X_hat = (X - mean) * 1.0 / np.sqrt(variance + eps)
        # scale anp shift
        out = gamma * X_hat + beta

    # 2d conv
    elif len(X.shape) == 4:
        # extract the dimensions
        N, C, H, W = X.shape
        print('N, C, H, W' ,N, C, H, W)
        # mini-batch mean
        mean = np.mean(X, axis=(0, 2, 3))
        print(mean, mean.shape)
        # mini-batch variance
        variance = np.mean((X - mean.reshape((1, C, 1, 1))) ** 2, axis=(0, 2, 3))
        # normalize
        X_hat = (X - mean.reshape((1, C, 1, 1))) * 1.0 / np.sqrt(variance.reshape((1, C, 1, 1)) + eps)
        # scale anp shift
        print(X_hat.shape)
        out = gamma.reshape((1, C, 1, 1)) * X_hat + beta.reshape((1, C, 1, 1))

    return out



B = np.array(range(6*2)).reshape((3,2,2,1))
ga = np.array([1,1])
be = np.array([0,0])


normed = pure_batch_norm(B, ga, be)
print(B)
print('-----')
print(normed)