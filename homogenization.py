
import numpy as np
import pandas as pd
import pickle
from keras.models import Model, Sequential
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Input, Dense, Flatten, Convolution3D, AveragePooling3D, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tensorflow.python.framework import ops
from keras.regularizers import l2
import h5py
import warnings
warnings.filterwarnings('ignore')

# load data
with open('./homogenization_data.pkl', 'rb') as f:
    data = pickle.load(f)
final_train_data = data['data']
final_train_label = data['label']
train_2p = data['data_2p']
######### normalize input data from (0,1) to (-0.5,0.5)
final_train_data = final_train_data - 0.5


print ('create model')
def build_model():
    inp = Input(shape=(51,51,51,1))

    x = Convolution3D(16, 3, 3, 3, init='glorot_normal', border_mode='same', dim_ordering='tf', W_regularizer=l2(0.001))(inp)
    x = Activation('relu')(x)
    x = AveragePooling3D(pool_size=(2, 2, 2))(x)
    x = Convolution3D(32, 3, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = AveragePooling3D(pool_size=(2, 2, 2))(x)
    x = Convolution3D(64, 3, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = AveragePooling3D(pool_size=(2, 2, 2))(x)
    x = Convolution3D(128, 3, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = AveragePooling3D(pool_size=(2, 2, 2))(x)
    x = Convolution3D(256, 3, 3, 3, init='glorot_normal', border_mode='same', W_regularizer=l2(0.001))(x)
    x = Activation('relu')(x)
    x = AveragePooling3D(pool_size=(2, 2, 2))(x)

    feature = Flatten()(x)
    model = Model(inp, feature)
    return model
print ('-------------------------')
print ('compile model')
model_img = build_model()
model_cor = build_model()
concate_feature = concatenate([model_img.output, model_cor.output])
x = Dense(2048, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(concate_feature)
x = Dense(1024, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)
prediction = Dense(1, init='glorot_normal', W_regularizer=l2(0.001))(x)
model = Model(input = [model_img.input, model_cor.input], output=prediction)
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
filepath = './my_model.hdf5'

print ('-------------------------')
print ('fit model')
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True)
history = model.fit([final_train_data, train_2p], final_train_label, batch_size=2, nb_epoch=2, validation_split=0.2, callbacks=[early_stopping,checkpoint])

