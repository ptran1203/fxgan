import warnings
warnings.filterwarnings('ignore',category=FutureWarning)
import keras
from keras.models import model_from_json, Sequential, Model
from keras.optimizers import Adam
from keras.layers import (Dense, Dropout, Activation, Flatten,
                        Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D,
                        Input, Add, ZeroPadding2D, BatchNormalization, Reshape, Concatenate)

# from keras.layers.core import Activation
import numpy as np

def convolutional_block(X, f, filters, stage, block, s = 2):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    F1, F2, F3 = filters
    
    X_shortcut = X

    # First component of main path 
    X = Conv2D(F1, (1, 1), strides = (s,s), name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    # Second component of main path
    X = Conv2D(F2, (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(F3, (1, 1), strides = (1,1), name = conv_name_base + '2c' )(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    ##### SHORTCUT PATH ####
    X_shortcut = Conv2D(F3, (1, 1), strides = (s,s), name = conv_name_base + '1')(X_shortcut)
    X_shortcut = BatchNormalization(axis = 3, name = bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X

def identity_block(X, f, filters, stage, block):
    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    
    # Retrieve Filters
    F1, F2, F3 = filters
    
    # Save the input value. You'll need this later to add back to the main path. 
    X_shortcut = X
    
    # First component of main path
    X = Conv2D(filters = F1, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2a')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    
    # Second component of main path
    X = Conv2D(filters = F2, kernel_size = (f, f), strides = (1,1), padding = 'same', name = conv_name_base + '2b')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path
    X = Conv2D(filters = F3, kernel_size = (1, 1), strides = (1,1), padding = 'valid', name = conv_name_base + '2c')(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2c')(X)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation
    X = Add()([X_shortcut, X])
    X = Activation('relu')(X)
    
    return X

def res_net50(input_shape, output_size):
    input_tensor = Input(input_shape)

    X = ZeroPadding2D((3, 3))(input_tensor)
    
    # Stage 1
    X = Conv2D(32, (7, 7), strides = (2, 2), name = 'conv1')(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [32, 32, 256], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [32, 32, 256], stage = 2, block = 'b')
    X = identity_block(X, 3, [32, 32, 256], stage = 2, block = 'c')

    # Stage 3 
    X = convolutional_block(X, f = 3, filters = [64, 64, 512], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 512], stage = 3, block = 'b')
    X = identity_block(X, 3, [64, 64, 512], stage = 3, block = 'c')
    X = identity_block(X, 3, [64,64, 512], stage = 3, block = 'd')

    # Stage 4
    X = convolutional_block(X, f = 3, filters = [128, 128, 1024], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 1024], stage = 4, block = 'b')
    X = identity_block(X, 3, [128, 128, 1024], stage = 4, block = 'c')
    X = identity_block(X, 3, [128, 128, 1024], stage = 4, block = 'd')
    X = identity_block(X, 3, [128, 128, 1024], stage = 4, block = 'e')
    X = identity_block(X, 3, [128, 128, 1024], stage = 4, block = 'f')

    # Stage 5 
    X = convolutional_block(X, f = 3, filters = [512, 512, 2048], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [256, 256, 2048], stage = 5, block = 'b')
    X = identity_block(X, 3, [256, 256, 2048], stage = 5, block = 'c')

    # AVGPOOL
    X = AveragePooling2D(pool_size=(2, 2), padding='same')(X)

    X = Flatten()(X)
    X = Dense(100, activation = 'relu')(X)
    X = Dense(output_size, activation='softmax')(X)
    model = Model(inputs = input_tensor, outputs = X, name='ResNet50')
    return model

def alex_net():
    # initialize alexNet model
    model = Sequential()
    # Convolutional layers
    model.add(Conv2D(kernel_size=(11, 11),
                    activation='relu',
                    strides=(4, 4),
                    filters=96,
                    padding='valid',
                    input_shape=(227,227,3)))
    model.add(MaxPooling2D(pool_size=(2, 2),
                        strides=(2, 2),
                        padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=256,
                    kernel_size=(5, 5),
                    strides=(1, 1),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                        strides=(2, 2),
                        padding='valid'))
    model.add(BatchNormalization())
    model.add(Conv2D(filters=384,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    activation='relu'))
    model.add(Conv2D(filters=384,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    activation='relu'))
    model.add(Conv2D(filters=256,
                    kernel_size=(3, 3),
                    strides=(1, 1),
                    padding='same',
                    activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2),
                        strides=(2, 2),
                        padding='valid'))
    model.add(Flatten())
    # fully connected
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.5))
    # output
    model.add(Dense(1, activation='linear'))
    model.compile(loss='mean_absolute_error',
                  optimizer=Adam(lr=1e-4),
                  metrics=['mse'])
    
    return model


if __name__ == '__main__':
    res = res_net50((128,128,3), 10)