import pickle
import tensorflow as tf 
from keras.regularizers import l2
from keras.models import Sequential, Model 
from keras.layers.core import Activation, Dense, Dropout
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import layers
from tensorflow.python.framework import ops 
import h5py
from sklearn.preprocessing import LabelBinarizer
import numpy as np 
from keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D, GlobalAveragePooling2D, concatenate, AveragePooling2D, BatchNormalization
import warnings
warnings.filterwarnings('ignore')

# load data
with open('./crystal_plasticity_data.pkl', 'rb') as f:
	data = pickle.load(f)
train_data = data['data']
train_label = data['label']
cor_data = data['data_2p']

labelencoder = LabelBinarizer()
labelencoder.fit(range(int(max(train_label))+1))
train_label = labelencoder.transform(train_label)

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


L2 = 0.00 # penalty for l2 regularization
bn = True # if using batch normalization
activation = 'relu' # activation function
pool = 0 # 0 for maxpooling, 1 for sum of max and average pooling
inp_size = (224, 224, 1) # input shape
# create 2D CNN model
print ('create model')
def build_model():
	inp = Input(shape=inp_size)
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

	x = residual_pool_changeChannelnum(32, 4, x, L2, bn, activation, pool)
	x = residual_pool_changeChannelnum(64, 4, x, L2, bn, activation, pool)
	feature_vector = GlobalAveragePooling2D()(x)
	model = Model(input=inp, output= feature_vector)
	return model
print ('-------------------------')
print ('fit model')
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
model.compile(loss = "categorical_crossentropy", optimizer='adam', metrics=["accuracy"])
filepath = './my_model.hdf5'
early_stopping = EarlyStopping(monitor='val_loss', patience=10)
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True)
history = model.fit([train_data, cor_data], train_label, batch_size=2, nb_epoch=10, validation_split=0.2, callbacks=[early_stopping, checkpoint])
