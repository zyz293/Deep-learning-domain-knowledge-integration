# 3DCNN, use homogenization_data.mat train with its autocorrelation of 2p
# convert regression problem

import numpy as np
import pandas as pd
import pickle
from keras import regularizers
from keras.models import Model, Sequential
from keras.optimizers import SGD, Adam, RMSprop
from keras.layers.core import Activation, Dense, Dropout
from keras.layers import Input, Dense, Flatten, Convolution3D, MaxPooling3D, AveragePooling3D, ZeroPadding3D, UpSampling2D, concatenate
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.callbacks import Callback
import tensorflow as tf
# tf.python.control_flow_ops = tf
from tensorflow.python.framework import ops
import scipy.io
import random
import scipy.stats
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from math import sqrt
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from keras.callbacks import LearningRateScheduler
from keras.regularizers import l2
import h5py
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import shuffle

path = '/raid/zyz293/MKS_contrast50/'

print 'load original data'
traindata = scipy.io.loadmat(path+'homogenization_train.mat')
# traindata = h5py.File(path+'homogenization_train.mat', 'r')
Mtrain = np.array(traindata['Mtrain'])
Cefftrain = np.array(traindata['Cefftrain'])
del traindata

data = h5py.File(path+'homogenization_train_2p.hdf5', 'r')
train_2p = np.array(data['train_2p'])
del data
"""
data = 
[('Mtrain', (5700, 132651), 'int'),
 ('Cefftrain', (5700, 1), 'single'),
 ('Mtest', (2850, 132651), 'int'),
 ('Cefftest', (2850, 1), 'single')]
"""

print 'change data to input format'
train_data = []
train_label = []
for i in range(len(Mtrain)):
    train_label.append(Cefftrain[i])
    temp = np.array(Mtrain[i])
    temp = temp.reshape(51,51,51,1)
    train_data.append(temp)
train_data = np.array(train_data)
print train_data.shape
print 'train data done'

train_label = np.array(train_label)
print train_label.shape
print 'train label done'
del Mtrain
del Cefftrain

final_train_data = []
final_train_label = []
final_train_data = np.array(train_data)
final_train_label = np.array(train_label)


######### normalize input data from (0,1) to (-0.5,0.5)
del train_data
del train_label
final_train_data = final_train_data - 0.5
# final_train_data = final_train_data * train_2p
final_train_data, train_2p, final_train_label = shuffle(final_train_data, train_2p, final_train_label, random_state=0)
######################
print 'training data shape:', final_train_data.shape
print 'training label shape: ', final_train_label.shape



# create 2D CNN model
print 'create model'
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

    # x = Dense(2048, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(feature)
    # # model.add(Dropout(0.8))
    # x = Dense(1024, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)
    # x = Dense(1, init='glorot_normal', W_regularizer=l2(0.001))(x)
    # model = Model(inp, x)
    return model
print '-------------------------'
experiment_num = 'homo_3DCNNClassify_regression_48'
n_epoch=2000

# compile model
print 'compile model'
model_img = build_model()
model_cor = build_model()
concate_feature = concatenate([model_img.output, model_cor.output])
x = Dense(2048, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(concate_feature)
x = Dense(1024, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)
prediction = Dense(1, init='glorot_normal', W_regularizer=l2(0.001))(x)
model = Model(input = [model_img.input, model_cor.input], output=prediction)
# model = build_model()
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mean_absolute_error'])
print model.summary()
filepath = '/raid/zyz293/MKS_contrast50/weights/'+experiment_num+'_bestweights.hdf5'

print '-------------------------'
print 'fit model'
early_stopping = EarlyStopping(monitor='val_loss', patience=20)
checkpoint = ModelCheckpoint(filepath, monitor='val_mean_absolute_error', verbose=1, save_best_only=True)
history = model.fit([final_train_data, final_train_data], final_train_label, batch_size=32, nb_epoch=n_epoch, validation_split=0.33, callbacks=[early_stopping,checkpoint])

# model.save_weights(filepath, overwrite=True)

# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['mean_absolute_error'])
plt.plot(history.history['val_mean_absolute_error'])
plt.title('MAE')
plt.ylabel('MAE')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./result_plot/'+experiment_num+'_model_metrics.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('./result_plot/'+experiment_num+'_model_loss.png')
plt.clf()
print '--------------------------'
print 'predict data'

testdata = scipy.io.loadmat(path+'homogenization_test.mat')
Mtest = np.array(testdata['Mtest'])
Cefftest = np.array(testdata['Cefftest'])
test_data = []
test_label = []
for i in range(len(Mtest)):
    test_label.append(Cefftest[i])
    temp = np.array(Mtest[i])
    temp = temp.reshape(51,51,51,1)
    test_data.append(temp)
test_data = np.array(test_data)
print test_data.shape
print 'test data done'

test_label = np.array(test_label)
print test_label.shape
print 'test label done'
del testdata
del Mtest
del Cefftest
final_test_data = []
final_test_label = []
final_test_data = np.array(test_data)
final_test_label = np.array(test_label)
mean_ceff_test = np.mean(final_test_label)
final_test_data = final_test_data - 0.5
del test_data
del test_label
print 'testing data shape: ', final_test_data.shape
print 'testing label shape: ', final_test_label.shape

data = h5py.File(path+'homogenization_test_2p.hdf5', 'r')
test_2p = np.array(data['test_2p'])
# final_test_data = final_test_data * test_2p

del data
del model
model_img = build_model()
model_cor = build_model()
concate_feature = concatenate([model_img.output, model_cor.output])
x = Dense(2048, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(concate_feature)
x = Dense(1024, init='glorot_normal', activation='relu', W_regularizer=l2(0.001))(x)
prediction = Dense(1, init='glorot_normal', W_regularizer=l2(0.001))(x)
model = Model(input = [model_img.input, model_cor.input], output=prediction)
# model = build_model()
model.load_weights(filepath)
final_pred_y = np.array(model.predict([final_test_data, final_test_data]))
mae = mean_absolute_error(final_test_label, final_pred_y)

sess = tf.Session()
print '------------------------'
print experiment_num
print 'MAE: ', mae
print 'MASE: ', mae / mean_ceff_test * 100
print "mean ceff: ", mean_ceff_test
# score = model.evaluate(final_test_data, final_test_label, batch_size=128)
# print 'socre: ', score



