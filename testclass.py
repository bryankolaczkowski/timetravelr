#!/usr/bin/env python3

import sklearn.model_selection
import numpy as np
import tensorflow as tf

# read data
with np.load('data.npz') as dict:
  data = dict['data']
  labl = dict['labl']

# slice off the end of the data
slice = 64
data = data[:,0:slice,:]

# train_validate split
(train_x,
valid_x,
train_y,
valid_y) = sklearn.model_selection.train_test_split(data, labl, train_size=0.9)

# dataset package
batchsize  = 64
train_data = tf.data.Dataset.from_tensor_slices((train_x,train_y))
valid_data = tf.data.Dataset.from_tensor_slices((valid_x,valid_y))
train_data = train_data.shuffle(buffer_size=train_x.shape[0]).batch(batchsize)
valid_data = valid_data.shuffle(buffer_size=valid_x.shape[0]).batch(batchsize)
print(train_data, valid_data)

# build model
repdim = 64
nheads = 2
keydim = repdim // 2
nblcks = 4
# input
inp = tf.keras.Input(shape=train_x.shape[1:])
out = tf.keras.layers.Dense(units=repdim)(inp)
for i in range(nblcks):
  # attention
  nrm = tf.keras.layers.LayerNormalization()(out)
  mha = tf.keras.layers.MultiHeadAttention(num_heads=nheads,
                                           key_dim=keydim,
                                           value_dim=repdim)(nrm,nrm,nrm)
  res = tf.keras.layers.Add()([out,mha])
  # feedforward
  nrm = tf.keras.layers.LayerNormalization()(res)
  ffa = tf.keras.layers.Dense(units=repdim,
                            activation=tf.keras.activations.relu)(nrm)
  ffb = tf.keras.layers.Dense(units=repdim)(ffa)
  out = tf.keras.layers.Add()([res,ffb])
# decision
out = tf.keras.layers.Flatten()(out)
out = tf.keras.layers.Dense(units=1,
                            activation=tf.keras.activations.sigmoid)(out)
model = tf.keras.Model(inputs=inp, outputs=out)
model.summary()

# compile model
model.compile(optimizer=tf.keras.optimizers.Nadam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])

# fit model
model.fit(train_data, epochs=20, validation_data=valid_data)
