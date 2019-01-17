#!/usr/bin/env python3
import sys 
import numpy as np
from scipy.io.wavfile import read
import pyhtk
import ipdb
import pandas as pd
import tensorflow as tf
np.set_printoptions(precision=5)

MEAN = -6.454230e+00
VAR = 1.359600e+07
ACT_FUN = 'relu'
CONTEXT = 3350
FILTER_LENGTH = 400
STRIDE = 10 
NUM_FILTERS = 64
SHOW_NUM_KERNELS = 10
SHOW_NUM_WEIGHTS = 10
TRAIN_DATA_SCP_PATH='/home/dawna/pwv20/work/chime4/exp/DR3/dnn-ce-base-dummy-set-up/lib/flists/train.scp'

uttName=sys.argv[1]
mmfFile=sys.argv[2]

def readInTrainScp(train_data_scp_path=TRAIN_DATA_SCP_PATH):
    uttWavFilePathDict={}
    trainSCPFile=open(train_data_scp_path)
    trainSCPLines=trainSCPFile.readlines()
    trainSCPFile.close()
    for line in trainSCPLines:
        items = line.strip().split('=')
        uttWavFilePathDict[items[0]] = items[1].split('[')[0]
    return uttWavFilePathDict

def get_normalized_audio_data(uttName, uttWavFilePathDict, mean=MEAN, var=VAR):
    wavFilePath=uttWavFilePathDict[uttName]
    rate, audio_data = read(wavFilePath)
    audio_data = np.asarray(audio_data, dtype=np.float32)
    audio_data -= mean
    audio_data /= (var**(0.5))
    return audio_data


def perform_cnn_forward_without_bias(audio_data, kernel_weights, context_input_len=CONTEXT, stride_len=STRIDE, kernel_len=FILTER_LENGTH, num_kernels=NUM_FILTERS):
    input_vector = audio_data[:context_input_len]
    output_num = int((context_input_len - kernel_len) / stride_len) + 1
    output = np.zeros((output_num, num_kernels))
    for output_idx in range(output_num):
        start_pos = output_idx * stride_len
        output_vector = kernel_weights.dot(input_vector[start_pos:start_pos+kernel_len])
        output[output_idx][:] = output_vector
    return output

def pretty_print_vector(vector, num_to_show):
    print(pd.DataFrame(data=vector[:num_to_show]).transpose())

def pretty_print_matrix(matrix, col_to_show, row_to_show):
    print(pd.DataFrame(data=(matrix[:col_to_show,:row_to_show])))

def print_cnn_output(cnn_output, weight, audio_data):
        print('------------------------------' + '\n')
        print('Show result for cnn output for {} inputs'.format(SHOW_NUM_KERNELS))
        for i in range(int(numPrintOutput)):
            start_pos = i * STRIDE
            input_vector = audio_data[start_pos:start_pos+FILTER_LENGTH]
            pretty_print_vector(input_vector, SHOW_NUM_KERNELS)
        print('Show result for cnn for {} kernels (col) and {} weights (row)'.format(SHOW_NUM_KERNELS, SHOW_NUM_WEIGHTS))
        pretty_print_matrix(weight, SHOW_NUM_KERNELS, SHOW_NUM_WEIGHTS)
        print('Show result for cnn output for {} kernels'.format(SHOW_NUM_KERNELS))
        pretty_print_matrix(cnn_output, numPrintOutput, 64)
        print('------------------------------' + '\n')

def get_layer_weight(mmfFile, layerName):
    hmmSet = pyhtk.HTKModelReader(mmfFile,'').getHiddenMarkovModelSet()
    return hmmSet.getNMatrixTable()[layerName].getValuesAsNumPyArray()

def make_input_batch(audio_data, network_shift=160, input_context=3350, batch_len=100):
    batch = np.zeros((batch_len, input_context))
    for i in range(batch_len):
        start_idx = i * network_shift 
        end_idx = start_idx + input_context
        batch[i][:] = audio_data[start_idx:end_idx]
    return batch

def calculate_convolution_with_tf(batch_input, kernel_weights, padding='VALID', strides=[10]):
    batch_input_shape = (batch_input.shape[0], batch_input.shape[1], 1)
    kernel_weights_shape = (kernel_weights.shape[0], 1, kernel_weights.shape[1])
    batch_input_fit = batch_input.reshape(batch_input_shape)
    kernel_weights_fit = kernel_weights.reshape(kernel_weights_shape)
    batch_input_tensor = tf.constant(batch_input_fit, dtype=tf.float32)
    kernel_weights_tensor = tf.constant(kernel_weights_fit, dtype=tf.float32)
    batch_output_tensor = tf.nn.convolution(input=batch_input_tensor, filter=kernel_weights_tensor, padding=padding, strides=strides)
    with tf.Session() as sess:
        output = sess.run(batch_output_tensor)
    return output

uttWavFilePathDict=readInTrainScp()
audio_data = get_normalized_audio_data(uttName, uttWavFilePathDict)
kernel_weights = get_layer_weight(mmfFile, 'CNNLayer1_kernels')
batch_input = make_input_batch(audio_data)
output = calculate_convolution_with_tf(batch_input, kernel_weights.transpose())
print('input')
pretty_print_matrix(batch_input, 4, 4) 

print('weights')
pretty_print_matrix(kernel_weights, 4, 4)

print('output')
pretty_print_matrix(output[0][:][:], 4, 4)

