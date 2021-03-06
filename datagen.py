#!/usr/bin/env python3

import sys
import numpy as np

def genData(N, h, T=1, rng=None):
  """
  brownian motion sim
  :param   int N : number of discrete time steps
  :param float h : standard deviations of increments (shape is dim of sim)
  :param   int T : number of continuous time steps
  :param     rng : random number generator
  """
  # default rng
  if not rng:
    rng = np.random.default_rng()
  # sim
  dt = 1. * T/N
  incs = rng.normal(0.0, h, (N,)+h.shape) * np.sqrt(dt)
  data = np.cumsum(incs, axis=0, dtype=np.float32)
  return data


if __name__ == '__main__':
  import matplotlib.pyplot as plt

  # set random seed
  rng = np.random.default_rng()
  if len(sys.argv) > 1:
    rng = np.random.default_rng(int(sys.argv[1]))

  ndata = 256000
  data_list = []
  labl_list = []
  for i in range(ndata):
    # generate data
    N = 128  # number of total time steps
    S =  32  # number of individual 'samples'
    h = rng.uniform(0.5, 3.0, S)
    data = genData(N, h, rng=rng)

    ## calculate label
    label = 0
    # split data into first and second halves
    frsthalf = data[    :N//2,:]
    scndhalf = data[N//2:,    :]
    # calculate correlations
    frstcorr = np.sum(np.triu(np.corrcoef(frsthalf, rowvar=False))) - S
    scndcorr = np.sum(np.triu(np.corrcoef(scndhalf, rowvar=False))) - S
    # calculate 'case'
    corrcutoff = 1.0
    if frstcorr < -corrcutoff and scndcorr > +corrcutoff:
      label = 1

    data_list.append(data)
    labl_list.append(label)

  # package data and labels
  data = np.stack(data_list)
  labl = np.array(labl_list, dtype=np.int32)
  print(data.shape, labl.shape)
  print('ones: {}/{} {:.4}'.format(np.sum(labl),
                                 labl.shape[0],
                                 np.sum(labl)/labl.shape[0]))

  # write data and labels to disk
  np.savez_compressed('./data.npz', data=data, labl=labl)

  # check data
  with np.load('./data.npz') as dfile:
    d = dfile['data']
    l = dfile['labl']
  print(d.shape, l.shape)

  # plot data
  fig = plt.figure(figsize=(16,8))
  plt.plot(data[0])
  plt.show()
