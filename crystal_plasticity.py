# use original data (resize by scipy) and use 24 crops, 
# train Residual modular CNN model from scratch. 
# train one ResNet for original image, one for its 2-point correlation matrix, 
# then concatenate to FC.

import tensorflow as tf 
from keras import optimizers
from keras.optimizers import SGD, Adam, RMSprop
from keras.regularizers import l2
from keras.models import Sequential, Model 
from keras.layers.core import Activation, Dense, Dropout
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping
from keras import regularizers
from keras import layers
from tensorflow.python.framework import ops 
from sklearn.metrics import accuracy_score
import h5py
from sklearn.utils import shuffle
from sklearn.preprocessing import LabelBinarizer
import numpy as np 
import matplotlib
matplotlib.use('Agg')
import pylab as plt
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate, UpSampling2D, AveragePooling2D, BatchNormalization
import sys

# load data
data = h5py.File('/raid/zyz293/NIST_data/NIST_train_resize_scipy_crop24.hdf5')
# train_data = np.array(data['train_data'])
# train_data = train_data[:,:,:,np.newaxis]
train_label = np.array(data['train_label'])
del data
data = h5py.File('/raid/zyz293/NIST_data/NIST_train_resize_scipy_auto2pOfcrop24.hdf5')
cor_data = np.asarray(data['train_data'])
cor_data = cor_data[:,:,:,np.newaxis]
del data

labelencoder = LabelBinarizer()
labelencoder.fit(range(int(max(train_label))+1))
train_label = labelencoder.transform(train_label)
# print train_data.shape
print cor_data.shape
print train_label.shape
# train_data = train_data + cor_data
# train_data, cor_data, train_label = shuffle(train_data, cor_data, train_label, random_state=0)
cor_data, train_label = shuffle(cor_data, train_label, random_state=0)

def residual_pool_changeChannelnum(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	first residual block, the input dimension of which is changed
	use 1*1 conv to match the dimension of channels of previous block.
	then have residual block
	then have a pooling layer
	BN: batch normalization (true or false)
	"""
	identity = Conv2D(num_filter, (1, 1), padding='same', W_regularizer=l2(L2))(prev_layer)
	if BN:
		identity = BatchNormalization(axis=-1)(identity)
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:
		z = BatchNormalization(axis=-1)(z)	
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		z = layers.add([a, b])
	else:
		z = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)	
	return z

def residual_pool(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	build residual block
	then have a pooling layer
	"""
	identity = prev_layer
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:
		z = BatchNormalization(axis=-1)(z)
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
		z = layers.add([a, b])
	else:
		z = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(z)
	return z

def residual_nopool_changeChannelnum(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	first residual block, the input dimension of which is changed
	use 1*1 conv to match the dimension of channels of previous block.
	then have residual block
	"""
	identity = Conv2D(num_filter, (1, 1), padding='same', W_regularizer=l2(L2))(prev_layer)
	if BN:	
		identity = BatchNormalization(axis=-1)(identity)
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:	
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:	
		z = BatchNormalization(axis=-1)(z)	
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	return z

def residual_block(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	build residual block
	"""
	identity = prev_layer
	z = prev_layer
	for i in range(num_layers-1):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:	
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:	
		z = BatchNormalization(axis=-1)(z)
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	return z

def residual_convpool_changeChannelnum(num_filter, num_layers, prev_layer, L2, BN, activation, pool):
	"""
	first residual block, the input dimension of which is changed
	use average pooling to match feature map size
	use 1*1 conv to match the dimension of channels.
	use conv with strides 2 to replace pooling	
	pad: when the size is even use 'same', otherwise 'valid'
	"""
	# identity = AveragePooling2D(pool_size=(2,2), strides=2, padding='same')(prev_layer)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(prev_layer)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(prev_layer)
		identity = layers.add([a, b])
	else:
		identity = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(prev_layer)
	identity = Conv2D(num_filter, (1, 1), padding='same', W_regularizer=l2(L2))(identity)
	if BN:	
		identity = BatchNormalization(axis=-1)(identity)
	z = Conv2D(num_filter, (3, 3), strides=(2,2), padding='same', W_regularizer=l2(L2))(prev_layer)
	if BN:	
		z = BatchNormalization(axis=-1)(z)
	z = Activation(activation)(z)
	for i in range(num_layers-2):
		z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
		if BN:	
			z = BatchNormalization(axis=-1)(z)
		z = Activation(activation)(z)
	z = Conv2D(num_filter, (3, 3), padding='same', W_regularizer=l2(L2))(z)
	if BN:	
		z = BatchNormalization(axis=-1)(z)
	z = layers.add([z, identity])
	z = Activation(activation)(z)
	return z


## parameter sets
experiment_num = 'scratch_365'
batchsize = 32
n_epoch = 2000
patience = 20
L2 = 0.00
# lr = 0.001
bn = True
activation = 'relu'
pool = 0 # 0 for maxpooling, 1 for sum of max and average pooling
crop_size = 24
inp_size = (224, 224, 1) # input shape
# create 2D CNN model
print 'create model'
def build_model():
	inp = Input(shape=inp_size)
	# x = Conv2D(64, (7, 7), strides=(2,2), padding='same', W_regularizer=l2(L2))(inp)
	x = Conv2D(16, (3, 3), padding='same', W_regularizer=l2(L2))(inp)
	if bn:
		x = BatchNormalization(axis=-1)(x)
	x = Activation(activation)(x)
	if pool:
		a = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
		b = AveragePooling2D(pool_size=(3, 3), strides=2, padding='same')(x)
		x = layers.add([a, b])
	else:
		x = MaxPooling2D(pool_size=(3, 3), strides=2, padding='same')(x)

# residual_pool_changeChannelnum: change # of channels of prev_layer and have pool at end
# residual_pool: just have pool at end
# residual_nopool_changeChannelnum: change # of channels of prev_layer and no pool
# residual_block: regular residual block
# residual_convpool_changeChannelnum: change # of channels of prev_layer and use conv to replace pool

	x = residual_pool_changeChannelnum(32, 4, x, L2, bn, activation, pool)
	# x = residual_block(32, 2, x, L2, bn, activation, pool)
	# x = residual_block(32, 2, x, L2, bn, activation, pool)
	# x = residual_pool(32, 2, x, L2, bn, activation, pool)
	x = residual_pool_changeChannelnum(64, 4, x, L2, bn, activation, pool)
	# x = residual_block(64, 2, x, L2, bn, activation, pool)
	# x = residual_pool(64, 2, x, L2, bn, activation, pool)
	# x = residual_nopool_changeChannelnum(128, 2, x, L2, bn, activation, pool)
	# x = residual_block(64, 2, x, L2, bn, activation, pool)
	# x = residual_pool(128, 2, x, L2, bn, activation, pool)
	# x = residual_pool_changeChannelnum(512, 2, x, L2, bn, activation, pool)
	# x = residual_pool(128, 2, x, L2, bn, activation, pool)
	# x = residual_pool_changeChannelnum(128, 2, x, L2, bn, activation, pool)
	# x = residual_pool(256, 2, x, L2, bn, activation, pool)
	# x = residual_pool(512, 2, x, L2)
	# x = residual_pool_changeChannelnum(256, 2, x, L2, bn, activation, pool)
	# x = residual_pool_changeChannelnum(512, 2, x, L2, bn, activation, pool)
	
	### fully conv
	feature_vector = GlobalAveragePooling2D()(x)
	model = Model(input=inp, output= feature_vector)
	### FC
	# x = Flatten()(x)
	# x = Dense(256, init='glorot_normal', W_regularizer=l2(L2))(feature_vector)
	# x = BatchNormalization()(x)
	# x = Activation(activation)(x)
	# x = Dropout(0.3)(x)
	# x = Dense(128, init='glorot_normal', W_regularizer=l2(L2))(x)
	# x = BatchNormalization()(x)
	# x = Activation(activation)(x)
	# x = Dropout(0.2)(x)
	# prediction = Dense(3, init='glorot_normal', activation='softmax', W_regularizer=l2(L2))(x)

	# compile the model 
	# model = Model(input=inp, output= prediction)
	# sgd = SGD(lr=lr, decay=lr/n_epoch, momentum=0.9, nesterov=True)
	# model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
	return model
print '-------------------------'
print 'fit model'
model_img = build_model()
model_cor = build_model()
concate_feature = concatenate([model_img.output, model_cor.output])
x = Dense(256, init='glorot_normal', W_regularizer=l2(L2))(concate_feature)
x = BatchNormalization()(x)
x = Activation(activation)(x)
x = Dropout(0.3)(x)
x = Dense(128, init='glorot_normal', W_regularizer=l2(L2))(x)
x = BatchNormalization()(x)
x = Activation(activation)(x)
x = Dropout(0.2)(x)
prediction = Dense(3, init='glorot_normal', activation='softmax', W_regularizer=l2(L2))(x)
model = Model(input = [model_img.input, model_cor.input], output = prediction)
# model = build_model()
model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
print model.summary()
filepath = '/raid/zyz293/NIST_data/weights/'+experiment_num+'_bestweights.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience=patience)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)

### save best model
history = model.fit([cor_data, cor_data], train_label, batch_size=batchsize, nb_epoch=n_epoch, validation_split=0.2, callbacks=[early_stopping, checkpoint])
### save model from last iteration
# history = model.fit(train_data, train_label, batch_size=batchsize, nb_epoch=n_epoch, validation_split=0.2, callbacks=[early_stopping])
# model.save_weights(filepath, overwrite=True)

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('/raid/zyz293/NIST_data/result_plot/'+experiment_num+'_model_metrics.png')
plt.clf()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('/raid/zyz293/NIST_data/result_plot/'+experiment_num+'_model_loss.png')
plt.clf()

# prediction
del model
# del train_data
del train_label
del cor_data
# model = build_model()
model_img = build_model()
model_cor = build_model()
concate_feature = concatenate([model_img.output, model_cor.output])
x = Dense(256, init='glorot_normal', W_regularizer=l2(L2))(concate_feature)
x = BatchNormalization()(x)
x = Activation(activation)(x)
x = Dropout(0.3)(x)
x = Dense(128, init='glorot_normal', W_regularizer=l2(L2))(x)
x = BatchNormalization()(x)
x = Activation(activation)(x)
x = Dropout(0.2)(x)
prediction = Dense(3, init='glorot_normal', activation='softmax', W_regularizer=l2(L2))(x)
model = Model(input = [model_img.input, model_cor.input], output = prediction)
model.load_weights(filepath)
# load testing data
data = h5py.File('/raid/zyz293/NIST_data/NIST_test_resize_scipy_crop24.hdf5')
# test_data = np.array(data['test_data'])
# test_data = test_data[:,:,:,np.newaxis]
test_label = np.array(data['test_label'])
data = h5py.File('/raid/zyz293/NIST_data/NIST_test_resize_scipy_auto2pOfcrop24.hdf5')
cor_data = np.array(data['test_data'])
cor_data = cor_data[:,:,:,np.newaxis]
del data
# print test_data.shape
print cor_data.shape
print test_label.shape
# test_data = test_data + cor_data
pred_y = np.array(model.predict([cor_data, cor_data]))
pred_y = np.array([sum(pred_y[i:i+crop_size]) for i in range(0, len(pred_y), crop_size)])
pred_y = np.argmax(pred_y, axis=1)
true_label = []
for i in range(0, len(test_label), crop_size):
	temp = test_label[i:i+crop_size]
	if len(np.unique(temp)) == 1:
		true_label.append(temp[0])
	else:
		print 'error in code'
		sys.exit()
print pred_y.shape
del test_label
true_label = np.array(true_label)
print true_label.shape
acc = accuracy_score(np.array(true_label), pred_y)


sess = tf.Session()
print '------------------------'
print experiment_num
print 'testing accuracy: ', acc



