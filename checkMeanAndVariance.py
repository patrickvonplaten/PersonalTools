#!/usr/bin/env python3
import sys 
import numpy as np
from scipy.io.wavfile import read

MEAN = -6.454230e+00
VAR = 1.359600e+07

uttName=sys.argv[1]
sampleInputString=sys.argv[2]
# give input as '14,24,87'
uttWavFilePaths={}
sampleIndexToReadOut = [int(x) for x in sampleInputString.split(',')]

trainDataPath='/home/dawna/pwv20/work/chime4/exp/DR3/dnn-ce-base-dummy-set-up/lib/flists/train.scp'
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

print('Read samples from: {}'.format(exampleWavFilePath))
for sampleIndex in sampleIndexToReadOut:
    print("Value of {} normalized sample is: {:.9f}".format(sampleIndex, audioData[sampleIndex]))
