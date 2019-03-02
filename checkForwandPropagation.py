#!/usr/bin/env python3
import sys 
import numpy as np
from scipy.io.wavfile import read
import pyhtk
import ipdb
import pandas as pd
np.set_printoptions(precision=5)

MEAN = -6.454230e+00
VAR = 1.359600e+07
ACT_FUN = 'relu'
CONTEXT = 3350
FILTER_LENGTH = 400
STRIDE = 10 
NETWORKSHIFT = 160
NUM_FILTERS = 64
SHOW_NUM_KERNELS = 4
SHOW_NUM_WEIGHTS = 4

uttName=sys.argv[1]
mmfFile=sys.argv[2]
numPrintOutput=int(sys.argv[3])

trainDataPath='/home/dawna/pwv20/work/chime4/exp/DR3/dnn-ce-base-dummy-set-up/lib/flists/train.scp'


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0) # only difference

def perform_cnn_forward_without_bias(audioData, kernel_weights, context_input_len=CONTEXT, stride_len=STRIDE, kernel_len=FILTER_LENGTH, num_kernels=NUM_FILTERS):
    input_vector = audioData[:context_input_len]
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

def print_cnn_output(cnn_output, weight, audioData):
        print('------------------------------' + '\n')
        print('Show result for input')
        for i in range(int(numPrintOutput)):
            start_pos = i * NETWORKSHIFT 
            input_vector = audioData[start_pos:start_pos+CONTEXT]
            pretty_print_vector(input_vector, SHOW_NUM_KERNELS)
        print('Show result for weights')
        pretty_print_matrix(weight, SHOW_NUM_KERNELS, SHOW_NUM_WEIGHTS)
        print('Show result for cnn output for kernels')
        pretty_print_matrix(cnn_output, numPrintOutput, SHOW_NUM_KERNELS)
        print('------------------------------' + '\n')

uttWavFilePaths={}
trainSCPFile=open(trainDataPath)
trainSCPLines=trainSCPFile.readlines()
trainSCPFile.close()

for line in trainSCPLines:
    items = line.strip().split('=')
    uttWavFilePaths[items[0]] = items[1].split('[')[0]

exampleWavFilePath=uttWavFilePaths[uttName]

rate, audioData = read(exampleWavFilePath)
audioData = np.asarray(audioData, dtype=np.float32)
audioData -= MEAN
audioData /= (VAR**(0.5))

hmmSet = pyhtk.HTKModelReader(mmfFile,'').getHiddenMarkovModelSet()
#layer1Bias = hmmSet.getNVectorTable()['CNNLayer1_bias'].getValuesAsNumPyArray()
layer1Weight = hmmSet.getNMatrixTable()['CNNLayer1_kernels'].getValuesAsNumPyArray()

cnn_output = perform_cnn_forward_without_bias(audioData, layer1Weight)
print_cnn_output(cnn_output, layer1Weight, audioData)
