# use keras
from __future__ import absolute_import, division, print_function, unicode_literals

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# tensorflow related
import tensorflow as tf
from tensorflow.python.ops import array_ops
from tensorflow.keras import backend as K
import tensorflow.keras as keras

# general packages
import scipy.io as sio     ## read matlab file data
import h5py
import time                ## time performance test
import numpy as np
import matplotlib.pyplot as plt
import os
import random
import datetime
import timeit
import sys

print(len(sys.argv))
print(str(sys.argv))



# gpu detection
tf.debugging.set_log_device_placement(True)
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
print('Found GPU at: {}'.format(device_name))

# validate tf version
print(tf.__version__)

from RNN_Model import build_model_GRU_init
###########################################################################
# 1) Some initial settings
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
  try:
    tf.config.experimental.set_virtual_device_configuration(
        gpus[0],
        [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=12000)])
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Virtual devices must be set before GPUs have been initialized
    print(e)


nTotal = 10000 # 7812
nBatch = 5000# 3906# 5000
nRNN_Unit = 32
nInput =  5   # theta, T1, T2, TE, TR
nOutput = 3   # m, dm/dT1, dm/dT2
nSeq_Length = 1120
model_valid = build_model_GRU_init(
    rnn_units = nRNN_Unit,
    input_batch = nBatch,
    input_size = nInput, # RF, T1, T2, don't put in mxy(n-1)
    output_size = nOutput)

str1 = "./TrainingResults/RNN_EPG.hdf5"
model_valid.load_weights(str1)

B1 = np.linspace(0.8,1.2,int(sys.argv[1]))
Data_Dic = np.zeros(shape=(nTotal * len(B1), nSeq_Length, nInput),dtype="float32")
State_Input_Dic = np.zeros(shape=(nTotal * len(B1), 3),dtype="float32")
State_Input_Dic[:,2] = -1.0
print(State_Input_Dic.shape)

str1 = "./Code/Experiments/1_MRF_Dictionary/flipangle.mat"

traindata = sio.loadmat(str1)
RFtrain = traindata.get("flipangles") # flipangles
TE = traindata.get("TE")
TR = traindata.get("TR")

str1 = "./Code/Experiments/1_MRF_Dictionary/Dictionary_EPG.mat"
traindata = sio.loadmat(str1)
Dictionary = traindata.get("Dictionary") # flipangles
T1T2B1 = traindata.get("T1T2B1")
print(T1T2B1.shape)
# T1 = T1T2B1[:,0]
# T1 = T1[:,np.newaxis]
# T2 = T1T2B1[:,1]
# T2 = T2[:,np.newaxis]

T1 = np.linspace(np.log(2.1), np.log(5.0),100)
T2 = np.linspace(np.log(0.01), np.log(2.0),100)

T1, T2 = np.meshgrid(T1,T2)
T1 = np.reshape(T1, (nTotal, 1)) # [:,np.newaxis]
T2 = np.reshape(T2, (nTotal, 1)) # T2[:,np.newaxis]

print(T1.shape, T2.shape, T1.dtype, T2.dtype)

for ii in range(len(B1)):
  print(ii)
  Data_Dic[ii * nTotal:(ii+1) * nTotal,0:nSeq_Length,0] = np.squeeze(RFtrain) * B1[ii] # RF
  Data_Dic[ii * nTotal:(ii+1) * nTotal,:,1] = T1          # T1
  Data_Dic[ii * nTotal:(ii+1) * nTotal,:,2] = T2          # T2
  Data_Dic[ii * nTotal:(ii+1) * nTotal,:,3] = 0.0049226   # TE
  Data_Dic[ii * nTotal:(ii+1) * nTotal,:,4] = 0.0087552   # TR
dataset_valid = tf.data.Dataset.from_tensor_slices(({"input_1": Data_Dic, "input_2": State_Input_Dic}))
dataset_valid = dataset_valid.batch(nBatch, drop_remainder=True) # nTotal

# compute y_predict
tic = time.time()
y_predict = model_valid.predict(dataset_valid) # model.predict(x) # x is tensor, but y_predict is numpy
toc = time.time()
print("Predict Time is:",toc-tic)

tic = time.time()
y_predict = model_valid.predict(dataset_valid) # model.predict(x) # x is tensor, but y_predict is numpy
toc = time.time()
print("Predict Time is:",toc-tic)

timings = open("/home/oheide/mrstat/mrstat_4_gpu/benchmarks/rnn/timings.txt", mode="a")
timings.write(str(toc-tic) + "\n")
timings.close()
sio.savemat('./Code/Experiments/1_MRF_Dictionary/Dictionary_RNN.mat', {'Dictionary':y_predict, 'T1T2B1':T1T2B1})
