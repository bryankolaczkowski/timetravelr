#!/usr/bin/env python3

import sklearn.model_selection
import numpy as np
import tensorflow as tf

from spcnrm import SpecNorm

@tf.function(experimental_relax_shapes=True)
def lrsact(x, alpha=0.4):
  """
  2-sided 'leaky-rectified' linear activation
  scales x by alpha*x whenever |x| > (1-alpha)
  """
  v  = 1.0 - alpha
  b  = v * v
  # leaky-rectify positive values
  c = tf.math.greater(x, v)
  r = tf.where(c, alpha*x+b, x)
  # leaky-rectify negative values
  c = tf.math.less(r, -v)
  r = tf.where(c, alpha*r-b, r)
  return r


# read data
with np.load('data.npz') as dict:
  data = dict['data']
  labl = dict['labl']

# add location information to data
pos = 2.0
loc = np.linspace(-pos, +pos, data.shape[-2], dtype=np.float32)
loc = np.expand_dims(loc, axis=-1)
loc = np.broadcast_to(loc, shape=(data.shape[0],data.shape[1],1))
data = np.concatenate((loc,data), axis=-1)

# train_validate split
(train_x,
valid_x,
train_y,
valid_y) = sklearn.model_selection.train_test_split(data, labl, train_size=0.9)

# dataset package
batchsize  = 128
train_data = tf.data.Dataset.from_tensor_slices((train_x,train_y))
valid_data = tf.data.Dataset.from_tensor_slices((valid_x,valid_y))
train_data = train_data.shuffle(buffer_size=train_x.shape[0]).batch(batchsize)
valid_data = valid_data.shuffle(buffer_size=valid_x.shape[0]).batch(batchsize)
print(train_data, valid_data)

# calculate pos (1) and neg (0) counts
tot = train_y.shape[0]
pos = np.sum(train_y)
neg = tot - pos

## build model
# hyperparams
repdim = 16
nheads = 4
keydim = repdim // 2
nblcks = 4
mhdout = 0.4
# input encoding
inp = tf.keras.Input(shape=train_x.shape[1:])
out = tf.keras.layers.Dense(units=repdim)(inp)
for i in range(nblcks):
  # attention
  nrm = tf.keras.layers.LayerNormalization()(out)
  mha = tf.keras.layers.MultiHeadAttention(num_heads=nheads,
                                           key_dim=keydim,
                                           value_dim=repdim,
                                           dropout=mhdout)(nrm,nrm,nrm)
  res = tf.keras.layers.Add()([out,mha])
  # feedforward
  nrm = tf.keras.layers.LayerNormalization()(res)
  ffa = tf.keras.layers.Dense(units=repdim, activation=lrsact)(nrm)
  ffb = tf.keras.layers.Dense(units=repdim)(ffa)
  out = tf.keras.layers.Add()([res,ffb])
# decision
obiasinit = tf.keras.initializers.Constant(np.log([pos/neg]))
out = tf.keras.layers.Flatten()(out)
out = SpecNorm(tf.keras.layers.Dense(units=1,
                                     activation=tf.keras.activations.sigmoid,
                                     bias_initializer=obiasinit))(out)
model = tf.keras.Model(inputs=inp, outputs=out)
model.summary()

## compile model
# metrics
mets = [tf.keras.metrics.AUC(curve='ROC', name='roc'),
        tf.keras.metrics.AUC(curve='PR',  name='prc')]
# compilation
model.compile(optimizer=tf.keras.optimizers.Nadam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=mets)

## fit model
# class weights
w0 = (1.0 / neg) * (tot / 2.0)
w1 = (1.0 / pos) * (tot / 2.0)
cwts = {0:w0, 1:w1}
# fit
model.fit(train_data,
          epochs=1000,
          class_weight=cwts,
          validation_data=valid_data)
