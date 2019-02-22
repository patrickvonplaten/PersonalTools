#!/usr/bin/env python2

import sys
import numpy as np
from scipy.io.wavfile import read
import ipdb

SCP_FILE_PATH=sys.argv[1]
FRAME_SHIFT=25 # tried out 5/10/25/50/100/160/400/800
FRAME_SIZE=FRAME_SHIFT
CORR_THRESHOLD=0.8
MICROPHONE_ORDER=[5,4,6,1,3]

def getRawPathesForBabelId(devScpFilePath):
    devScpFile = open(devScpFilePath)
    devScpLines = devScpFile.readlines()
    devScpFile.close()
    utterancePaths = {}
    for line in devScpLines:
        items = line.strip().split('=')
        babelId = items[0].split('.')[0]
        rawPath = items[1].split('.')[0] + '.CH{}.wav'
        utterancePaths[babelId] = rawPath
    return utterancePaths

def getEnergyPerFramePerChannel(path, channel):
    return getEnergyPerFrame(path.format(channel))

def getEnergyPerFrame(path):
    rate, audioData = read(path)
    lenAudioData = len(audioData)
    numFrames = int((lenAudioData - FRAME_SIZE) / FRAME_SHIFT) + 1
    uttFrameEnergies=np.zeros(numFrames)
    for outputFrameIdx in range(numFrames):
        startFrame = outputFrameIdx * FRAME_SHIFT
        endFrame = startFrame + FRAME_SIZE
        squaredAudioData = audioData[startFrame:endFrame].astype('int64')**2
        uttFrameEnergies[outputFrameIdx] = np.sum(squaredAudioData)
    return uttFrameEnergies

def getEnergyCorrelation(energyChannel1, energyChannel2):
    return np.corrcoef(energyChannel1,energyChannel2)[1][0] 

def getAvgEnergyCorrelationOfChannel(path, channel, workingMicrophones):
    compareChannels = [ microphone for microphone in workingMicrophones if microphone != channel]
    channelEnergy =  getEnergyPerFramePerChannel(path, channel)
    avgChannelEnergyCorrelation = 0
    for channelToCompare in compareChannels: 
        channelToCompareEnergy = getEnergyPerFramePerChannel(path, channelToCompare)
        avgChannelEnergyCorrelation += getEnergyCorrelation(channelEnergy, channelToCompareEnergy)
    return avgChannelEnergyCorrelation / float(len(compareChannels))

def isChannelBrokenOfUtterancePath(path, channel):
    avgCorrelationOfUtterance = getAvgEnergyCorrelationOfChannel(path, channel, MICROPHONE_ORDER)
    if avgCorrelationOfUtterance < CORR_THRESHOLD:
        return True
    return False

def greedyGetChannelToDecode(path, MICROPHONE_ORDER):
    for microphone in MICROPHONE_ORDER:
        if(not isChannelBrokenOfUtterancePath(path, microphone)):
            return [microphone], ''
    return MICROPHONE_ORDER, ''

def printChannelToDecodePerUtterance():
    utterancePaths=getRawPathesForBabelId(SCP_FILE_PATH)
    for utteranceId in utterancePaths: 
        rawUtterancePath = utterancePaths[utteranceId]
        workingMicrophones, warning = removeBrokenMicrophones(rawUtterancePath, list(MICROPHONE_ORDER))
#        print(utteranceId + '=' + str(workingMicrophones[0]) + warning)
        print(utteranceId + '=' + str(workingMicrophones[0]))

def removeBrokenMicrophones(rawUtterancePath, microphones): 
    microphoneCorrCoefs = []
    if(len(microphones) < 3):
        return microphones, ' WARINING: 3 MICROPHONES HAVE CORRELATION < 0.8'
    for microphone in microphones: 
        microphoneCorrCoef = getAvgEnergyCorrelationOfChannel(rawUtterancePath, microphone, microphones)
        microphoneCorrCoefs.append(microphoneCorrCoef)
    if(min(microphoneCorrCoefs) < CORR_THRESHOLD):
        microphoneToRemoveIdx = microphoneCorrCoefs.index(min(microphoneCorrCoefs))
        del microphones[microphoneToRemoveIdx]
        return removeBrokenMicrophones(rawUtterancePath, microphones)
    return microphones, ''

if __name__ == "__main__":
    printChannelToDecodePerUtterance()
