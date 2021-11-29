#!/usr/bin/env python3

import sys
import distutils.util
import argparse

import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sklearn.model_selection
import numpy as np
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import Loss
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import Model

### DATA #######################################################################

def getRealData(datafile, trainlbl, trainsiz, batchsize, trainprop):
  # read data
  with np.load(datafile) as dict:
    data = dict['data']
    labl = dict['labl']

  # extract *only* the labels we want
  indices = np.argwhere(np.equal(labl, trainlbl)).flatten()
  data = np.take(data, indices, axis=0)
  labl = np.take(labl, indices, axis=0)
  print(data.shape, labl.shape)

  # train_validate split
  (train_x,
   valid_x,
   train_y,
   valid_y) = sklearn.model_selection.train_test_split(data,
                                                       labl,
                                                       train_size=trainprop)

  # cut training data down to size!
  if trainsiz:
    train_x = train_x[:trainsiz]
    train_y = train_y[:trainsiz]
  print(train_x.shape, train_y.shape, valid_x.shape, valid_y.shape)

  # dataset package
  train_data = tf.data.Dataset.from_tensor_slices(train_x)
  valid_data = tf.data.Dataset.from_tensor_slices(valid_x)
  train_data = train_data.shuffle(buffer_size=train_x.shape[0]).batch(batchsize)
  valid_data = valid_data.shuffle(buffer_size=valid_x.shape[0]).batch(batchsize)
  print(train_data, valid_data)

  train_n = train_x.shape[0]
  valid_n = valid_x.shape[0]

  return (train_x.shape[1], train_x.shape[2],
          train_n, valid_n,
          train_data, valid_data)

### GAN ########################################################################

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

class WassersteinLoss(Loss):
  """
  implements wasserstein loss function

  'earth mover' distance from:
    https://arxiv.org/pdf/1701.07875.pdf
    https://arxiv.org/pdf/1704.00028.pdf
  """
  def __init__(self):
    super(WassersteinLoss, self).__init__(name='wasserstein_loss')
    return

  def call(self, y_true, y_pred):
    return K.mean(y_true * y_pred)

class GanOptimizer(Optimizer):
  """
  implements a generator,discriminator optimizer pair
  """
  def __init__(self,
               gen_optimizer='sgd',
               dis_optimizer='sgd',
               **kwargs):
    super(GanOptimizer, self).__init__(name='GanOptimizer', **kwargs)
    self.gen_optimizer = tf.keras.optimizers.get(gen_optimizer)
    self.dis_optimizer = tf.keras.optimizers.get(dis_optimizer)
    return

  def apply_gradients(self, grads_and_vars,
                      name=None, experimental_aggregate_gradients=True):
    raise NotImplementedError('GAN optimizer should call '
                              'apply_generator_gradients and '
                              'apply_discriminator_gradients instead')

  def apply_generator_gradients(self, grads_and_vars):
    return self.gen_optimizer.apply_gradients(grads_and_vars)

  def apply_discriminator_gradients(self, grads_and_vars):
    return self.dis_optimizer.apply_gradients(grads_and_vars)

  def get_config(self):
    config = super(GanOptimizer, self).get_config()
    config.update({
      'gen_optimizer' : tf.keras.optimizers.serialize(self.gen_optimizer),
      'dis_optimizer' : tf.keras.optimizers.serialize(self.dis_optimizer),
    })
    return config


class GAN(Model):
  """
  generative adversarial network
  """
  def __init__(self,
               generator,
               discriminator,
               **kwargs):
    super(GAN, self).__init__(**kwargs)
    self.genr  = generator
    self.disr  = discriminator
    return

  def compile(self,
              optimizer=GanOptimizer(),
              loss=WassersteinLoss(),
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              steps_per_execution=None,
              **kwargs):
    super(GAN, self).compile(optimizer=optimizer,
                             loss=loss,
                             metrics=metrics,
                             loss_weights=loss_weights,
                             weighted_metrics=weighted_metrics,
                             run_eagerly=run_eagerly,
                             steps_per_execution=steps_per_execution,
                             **kwargs)
    return

  def call(self, inputs, training=None):
    """
    inputs should be gaussian random noise (in the right shape!)
    """
    gdta1 = self.genr(inputs, training=training)  # generate first  data
    gdta2 = self.genr(inputs, training=training)  # generate second data
    dsrsr = self.disr((gdta1, gdta2), training=training)
    return (gdta1, dsrsr)

  def _calc_loss(self, qry_data, gnr_data, y, training=None):
    """
    calculates appropriate loss function
    """
    y_hat = self.disr((qry_data, gnr_data), training=training)
    return self.compiled_loss(y, y_hat)

  def _get_step_setup(self, inputs):
    """
    returns positive and negative labels
    """
    bs    = tf.shape(inputs)[0]  # batch size
    pones =  tf.ones((bs,1))     # positive labels
    nones = -tf.ones((bs,1))     # negative labels
    return pones, nones

  def _get_rnd_geninput(self, inputs):
    """
    returns correctly shaped generator random input
    """
    return tf.random.normal(shape=tf.shape(inputs))

  def test_step(self, inputs):
    """
    single validation step; inputs are real data
    """
    pones, nones = self._get_step_setup(inputs)

    # discriminator loss on real data
    rnd = self._get_rnd_geninput(inputs)
    disr_rl = self._calc_loss(qry_data=inputs,
                              gnr_data=self.genr(rnd, training=False),
                              y=nones,
                              training=False)
    # discriminator loss on fake data
    rnd1 = self._get_rnd_geninput(inputs)
    rnd2 = self._get_rnd_geninput(inputs)
    disr_fk = self._calc_loss(qry_data=self.genr(rnd1, training=False),
                              gnr_data=self.genr(rnd2, training=False),
                              y=pones,
                              training=False)
    # generator loss
    rnd1 = self._get_rnd_geninput(inputs)
    rnd2 = self._get_rnd_geninput(inputs)
    genr_ls = self._calc_loss(qry_data=self.genr(rnd1, training=False),
                              gnr_data=self.genr(rnd2, training=False),
                              y=nones,
                              training=False)
    return {'disr_rl' : disr_rl,
            'disr_fk' : disr_fk,
            'genr_ls' : genr_ls,}

  def train_step(self, inputs):
    """
    single training step; inputs are real data
    """
    pones, nones = self._get_step_setup(inputs)

    # train discriminator using real data
    with tf.GradientTape() as tape:
      rnd = self._get_rnd_geninput(inputs)
      disr_rl = self._calc_loss(qry_data=inputs,
                                gnr_data=self.genr(rnd, training=False),
                                y=nones,
                                training=True)
    grds = tape.gradient(disr_rl, self.disr.trainable_weights)
    self.optimizer.apply_discriminator_gradients(zip(grds,
                                                 self.disr.trainable_weights))

    # train discriminator using fake data
    with tf.GradientTape() as tape:
      rnd1 = self._get_rnd_geninput(inputs)
      rnd2 = self._get_rnd_geninput(inputs)
      disr_fk = self._calc_loss(qry_data=self.genr(rnd1, training=False),
                                gnr_data=self.genr(rnd2, training=False),
                                y=pones,
                                training=True)
    grds = tape.gradient(disr_fk, self.disr.trainable_weights)
    self.optimizer.apply_discriminator_gradients(zip(grds,
                                                 self.disr.trainable_weights))

    # train generator
    with tf.GradientTape() as tape:
      rnd1 = self._get_rnd_geninput(inputs)
      rnd2 = self._get_rnd_geninput(inputs)
      genr_ls = self._calc_loss(qry_data=self.genr(rnd1, training=True),
                                gnr_data=self.genr(rnd2, training=False),
                                y=nones,
                                training=False)
    grds = tape.gradient(genr_ls, self.genr.trainable_weights)
    self.optimizer.apply_generator_gradients(zip(grds,
                                             self.genr.trainable_weights))

    return {'disr_rl' : disr_rl,
            'disr_fk' : disr_fk,
            'genr_ls' : genr_ls}

  def get_config(self):
    config = super(GAN, self).get_config()
    config.update({
      'generator'       : tf.keras.layers.serialize(self.genr),
      'discriminator'   : tf.keras.layers.serialize(self.disr),
    })
    return config

class GenEnc(tf.keras.layers.Layer):
  """
  initial encoding layer for generator
  """
  def __init__(self, *args, **kwargs):
    super(GenEnc, self).__init__(*args, **kwargs)
    return

  def call(self, inputs):
    """
    inputs should be gaussian random in data shape
    """
    dt = tf.math.reciprocal(tf.cast(tf.shape(inputs)[1], dtype=tf.float32))
    return tf.math.cumsum(inputs * tf.math.sqrt(dt), axis=1)

class ConcatPos(tf.keras.layers.Layer):
  """
  concatenates (prepends) position information to data
  """
  def __init__(self, timesteps, *args, **kwargs):
    super(ConcatPos, self).__init__(*args, **kwargs)
    self.timesteps = timesteps
    self.pos = tf.linspace(0.0, 1.0, self.timesteps)
    self.pos = tf.expand_dims(tf.expand_dims(self.pos, axis=-1), axis=0)
    return

  def call(self, inputs):
    bs = tf.shape(inputs)[0]
    ps = tf.tile(self.pos, multiples=(bs,1,1))  # sequence position encoding
    return tf.concat((ps, inputs), axis=-1)     # prepend position to inputs

  def get_config(self):
    config = super(ConcatPos, self).get_config()
    config.update({
      'timesteps' : self.timesteps,
    })
    return config

class PosMaskMHA(tf.keras.layers.Layer):
  """
  multi-head attention transformer block with position masking
  """
  def __init__(self, width, dim, heads, *args, **kwargs):
    super(PosMaskMHA, self).__init__(*args, **kwargs)
    self.width = width
    self.dim   = dim
    self.heads = heads
    # multi-head attention
    self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.heads,
                                                  key_dim=self.dim // 2)
    # feed-forward
    self.ff1 = tf.keras.layers.Dense(units=self.dim * 2, activation=lrsact)
    self.ff2 = tf.keras.layers.Dense(units=self.dim,     activation=lrsact)
    # layer normalization
    self.ln1 = tf.keras.layers.LayerNormalization()
    self.ln2 = tf.keras.layers.LayerNormalization()
    # position masking
    msk_list = [tf.zeros(shape=(1,self.width))]
    for i in range(1,self.dim):
      msk_list.append(tf.ones(shape=(1,self.width)))
    self.msk = tf.stack(msk_list, axis=-1)
    return

  def call(self, inputs):
    # sub-block 1 - pre-lyrnorm, multi-head attn, residual
    a = self.ln1(inputs)
    a = self.mha(a,a)
    a = inputs + (a * self.msk)
    # sub-block 2 - pre-lyrnorm, feed-forward, residual
    b = self.ln2(a)
    b = self.ff1(b)
    b = self.ff2(b)
    b = a + (b * self.msk)
    return b

  def get_config(self):
    config = super(PosMaskMHA, self).get_config()
    config.update({
      'width' : self.width,
      'dim'   : self.dim,
      'heads' : self.heads,
    })
    return config

class DoutPosMaskMHA(PosMaskMHA):
  """
  position-masked multi-head attention transformer with dropout
  """
  def __init__(self, doutr, *args, **kwargs):
    super(DoutPosMaskMHA, self).__init__(*args, **kwargs)
    self.doutr = doutr
    self.dout1 = tf.keras.layers.Dropout(rate=doutr)
    self.dout2 = tf.keras.layers.Dropout(rate=doutr)
    return

  def call(self, inputs):
    # sub-block 1 - pre-lyrnorm, multi-head attn, residual
    a = self.ln1(inputs)
    a = self.mha(a,a)
    a = self.dout1(a)
    a = inputs + (a * self.msk)
    # sub-block 2 - pre-lyrnorm, feed-forward, residual
    b = self.ln2(a)
    b = self.ff1(b)
    b = self.ff2(b)
    b = self.dout2(b)
    b = a + (b * self.msk)
    return b
    return

  def get_config(self):
    config = super(DoutPosMaskMHA, self).get_config()
    config.update({
      'doutr' : self.doutr,
    })
    return config

## generator build ##
def generator_build(outstps, outdim, nblcks=2, nheads=4):
  # create input, gaussian random noise in data shape
  inputs  = tf.keras.Input(shape=[outstps, outdim], name='rndin')
  # initial data encoding
  repdim = outdim * 2
  out = GenEnc(name='encod')(inputs)              # basic encoding
  out = tf.keras.layers.Dense(units=repdim)(out)  # project into repdim
  out = ConcatPos(outstps, name='postn')(out)     # add position information
  repdim += 1  # account for position information
  # transformation(s)
  for i in range(nblcks):
    out = PosMaskMHA(width=outstps, dim=repdim, heads=nheads)(out)
  # project into data dimension
  outputs = tf.keras.layers.Dense(units=outdim)(out)
  return Model(inputs=inputs, outputs=outputs)

## discriminator build ##
def discriminator_build(instps, indim, nblcks=2, nheads=4, doutrt=0.2):
  # create input
  in1 = tf.keras.Input(shape=[instps, indim], name='dtin1')
  in2 = tf.keras.Input(shape=[instps, indim], name='dtin2')
  out = tf.keras.layers.Concatenate(name='concat')([in1,in2])
  # initial encoding
  repdim = indim * 2
  out = tf.keras.layers.Dense(units=repdim)(out)  # projection
  out = ConcatPos(instps, name='postn')(out)      # add position
  repdim += 1  # account for position information
  # transformation(s)
  for i in range(nblcks):
    out = DoutPosMaskMHA(doutr=doutrt,
                         width=instps,
                         dim=repdim,
                         heads=nheads)(out)
  # flatten and score
  outputs = tf.keras.layers.Flatten(name='flatn')(out)
  outputs = tf.keras.layers.Dense(units=1, name='outpt')(outputs)
  return Model(inputs=[in1,in2], outputs=outputs)

class PlotCallback(tf.keras.callbacks.Callback):
  """
  plot generated data
  """
  def __init__(self, ex_inpt, log_dir='logs'):
    self.writer  = tf.summary.create_file_writer(log_dir + '/gen')
    self.ex_inpt = ex_inpt
    return

  def plot_data(self, data):
    fig = plt.figure(figsize=(8,4))
    plt.plot(data)
    return fig

  def plot_to_image(self, plot):
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close(plot)
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    image = tf.expand_dims(image, 0)
    return image

  def on_epoch_end(self, epoch, logs=None):
    # generate example datas
    (dta,scr) = self.model(self.ex_inpt)
    fig = self.plot_data(dta[0])
    img = self.plot_to_image(fig)
    with self.writer.as_default():
      tf.summary.image('GenData', img, step=epoch)
    return

### EXECUTIONS #################################################################

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
                description='generative adversarial network',
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  # data input
  group = parser.add_argument_group('data')
  group.add_argument('-f', '--file', dest='file',
                     help='numpy data file', metavar='data.npz')
  group.add_argument('--train_size', dest='train_size', type=int,
                     help='number of training samples', metavar='N')
  group.add_argument('--train_label', dest='train_label', type=int,
                     help='training label, 0|1', metavar='N')

  # model
  group = parser.add_argument_group('model')
  group.add_argument('--blocks', dest='blocks', type=int,
                     help='number of blocks', metavar='N')
  group.add_argument('--heads', dest='heads', type=int,
                     help='number of multiheads', metavar='N')
  group.add_argument('--dropout', dest='dropout', type=float,
                     help='dropout proportion', metavar='N')

  # training regime
  group = parser.add_argument_group('training')
  group.add_argument('--train_prop', dest='train_prop', type=float,
                     help='initial training data proportion', metavar='N')
  group.add_argument('--batch_size', dest='batch_size', type=int,
                     help='training batch size', metavar='N')
  group.add_argument('--epochs', dest='epochs', type=int,
                     help='number of training epochs', metavar='N')
  group.add_argument('--learning_rate', dest='learn_rate', type=float,
                     help='base learning rate for generator', metavar='N')
  group.add_argument('--learning_rate_mult', dest='learn_rate_mult',
                     type=float,
                     help='discriminator learning rate multiplier',
                     metavar='N')

  parser.set_defaults(file='data.npz',
                      train_size=2048,
                      train_label=0,

                      blocks=4,
                      heads=4,
                      dropout=0.4,

                      train_prop=0.9,
                      batch_size=256,
                      epochs=5000,
                      learn_rate=1.0e-5,
                      learn_rate_mult=0.1)

  args = parser.parse_args()


  (data_timesteps,
   data_dimension,
   train_data_samples,
   valid_data_samples,
   train_data,
   valid_data) = getRealData(args.file,
                             args.train_label,
                             args.train_size,
                             args.batch_size,
                             args.train_prop)

  generator = generator_build(data_timesteps,
                              data_dimension,
                              nblcks=args.blocks,
                              nheads=args.heads)
  discriminator = discriminator_build(data_timesteps,
                                      data_dimension,
                                      nblcks=args.blocks,
                                      nheads=args.heads,
                                      doutrt=args.dropout)
  generator.summary()
  discriminator.summary()

  # create optimizer
  steps_per_epoch = int(train_data_samples / args.batch_size)
  decay_steps     = steps_per_epoch * 50  # reduce lr every 50 epochs
  total_steps     = steps_per_epoch * args.epochs
  decay_rate      = (1.0e-7/args.learn_rate)**(decay_steps/total_steps)

  gsch = tf.keras.optimizers.schedules.ExponentialDecay(\
                  initial_learning_rate=args.learn_rate,
                  decay_steps=decay_steps,
                  decay_rate=decay_rate,
                  staircase=True)
  dsch = tf.keras.optimizers.schedules.ExponentialDecay(\
                  initial_learning_rate=args.learn_rate * args.learn_rate_mult,
                  decay_steps=decay_steps*2,
                  decay_rate=decay_rate,
                  staircase=True)
  gopt = tf.keras.optimizers.SGD(learning_rate=gsch,
                                 momentum=0.8,
                                 nesterov=True)
  dopt = tf.keras.optimizers.SGD(learning_rate=dsch,
                                 momentum=0.8,
                                 nesterov=True)
  opt  = GanOptimizer(gopt, dopt)

  # compile gan
  gan = GAN(generator, discriminator)
  gan.compile(optimizer=opt)

  # set up callbacks
  callbacks = []
  # tensorboard
  rng = np.random.default_rng()
  example_input = rng.normal(0.0, 1.0, (1,data_timesteps,data_dimension))
  tbdir = 'tblog'
  callbacks.append(tf.keras.callbacks.TensorBoard(log_dir=tbdir))
  callbacks.append(PlotCallback(example_input, log_dir=tbdir))

  # fit gan
  gan.fit(train_data,
          epochs=args.epochs,
          validation_data=valid_data,
          callbacks=callbacks)

  # save final generator model
  #gan.genr.save(genfname)

  sys.exit(0)
