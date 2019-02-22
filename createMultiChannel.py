#!/usr/bin/env python3
from scipy.io.wavfile import read, write
import sys
import os
import numpy as np
import ipdb
from scipy.io.wavfile import read, write

MULTICHANNEL_ID = 13456
BABEL_MULTICHANNEL_ID='55'
ABS_MULTICHANNEL_FOLDER='/home/dawna/pwv20/work/chime4/data/raw/multi_channel'
ABS_SINGLECHANNEL_FOLDER='/home/dawna/chime/import/data/CHiME4/data/audio/16kHz/'
RELATIV_FOLDER_TO_COPY = 2
DO_SAVE = False
CHANNEL_ID_TO_REPLACE = 5
SCPFILE_TO_EXTRACT=sys.argv[1]
SCPFILE_MULTI_CHANNEL=sys.argv[2]

def transformPathToWavFileToFormatScpFileName(scpPathToWavFile):
    utterancePath = scpPathToWavFile.split('/')[-1]
    utteranceId = utterancePath.split('.')[0]
    utteranceInterval = utterancePath.split('[')[-1]
    return utteranceId + '.CH{}.wav[' + utteranceInterval

def getRelativPath(scpPathToWavFile):
    return '/'.join(scpPathToWavFile.split('/')[-1-RELATIV_FOLDER_TO_COPY:-1])

def transformToMultiChannelScpFileName(scpPathToWavFile):
    formatScpFileName = transformPathToWavFileToFormatScpFileName(scpPathToWavFile)
    relativePath = getRelativPath(scpPathToWavFile)
    return os.path.join(ABS_MULTICHANNEL_FOLDER, relativePath, formatScpFileName.format(12345))

def transformToFormatBabelUtteranceName(babelUtteranceName, channelIdToReplace):
    babelUtteranceNameItems = babelUtteranceName.split('-0' + str(channelIdToReplace) + '_')
    return babelUtteranceNameItems[0] + '-{}_' + babelUtteranceNameItems[1]
    
def createFormatDictionaryAndList(scpFilePath, channelIdToReplace=CHANNEL_ID_TO_REPLACE):
    scpFile = open(scpFilePath)
    lines = scpFile.readlines()
    scpFile.close()
    scpDict = {}
    utteranceIdList = []
    for line in lines:
        lineItems = line.strip().split('=')
        formatBabelUtteranceName = transformToFormatBabelUtteranceName(lineItems[0], channelIdToReplace)
        relativPath = getRelativPath(lineItems[1])
        formatScpFileName = transformPathToWavFileToFormatScpFileName(lineItems[1])
        scpDict[formatBabelUtteranceName] = os.path.join(relativPath, formatScpFileName)
        utteranceIdList.append(formatBabelUtteranceName)
    return scpDict, utteranceIdList

def getAbsPathToWavFile(baseFolder, dictValueFileName, channelId):
    return os.path.join(baseFolder, dictValueFileName.format(channelId).split('[')[0])

def writeMultiChannelScpFile(scpDict, utteranceIdList, scpFilePathMultiChannel=SCPFILE_MULTI_CHANNEL):
    with open(scpFilePathMultiChannel, 'w+') as scpFileMultiChannel:
        for key in utteranceIdList:
            lineToWrite = key.format(BABEL_MULTICHANNEL_ID) + '=' + os.path.join(ABS_MULTICHANNEL_FOLDER, scpDict[key].format(MULTICHANNEL_ID)) + '\n'
            scpFileMultiChannel.write(lineToWrite)

def saveMultiChannelWavFiles(scpDict, utteranceIdList):
    channelsToExtract = [1,3,4,5,6]
    numChannels = len(channelsToExtract)
    numUtterances = len(utteranceIdList)
    for uttCount, utteranceId in enumerate(utteranceIdList):
        scpFileFormatId = scpDict[utteranceId]
        numSamples = None
        for channelIdx, channelToExtract in enumerate(channelsToExtract):
            filePathToExtract = os.path.join(ABS_SINGLECHANNEL_FOLDER ,scpFileFormatId)
            sampleRate, audioData = read(getAbsPathToWavFile(ABS_SINGLECHANNEL_FOLDER, scpFileFormatId, channelToExtract))
            if(channelToExtract == 1):
                numSamples = len(audioData)
                multiChannelData = np.zeros((numSamples, numChannels), dtype=np.int16)
            multiChannelData[:,channelIdx] = audioData
            assert len(audioData) == numSamples, '{} and {} have different number of samples'.format(filePathToExtract.format(1), filePathToExtract.format(channelToExtract)) 
        multiChannelUtteranceFilePath = getAbsPathToWavFile(ABS_MULTICHANNEL_FOLDER,scpFileFormatId, MULTICHANNEL_ID)
        multiChannelUtteranceDir = '/'.join(multiChannelUtteranceFilePath.split('/')[:-1])
        if not os.path.exists(multiChannelUtteranceDir):
            os.makedirs(multiChannelUtteranceDir)
        if not os.path.isfile(multiChannelUtteranceFilePath):
            write(multiChannelUtteranceFilePath, sampleRate, multiChannelData)
        print('Finished {} % ...'.format(100 * float(uttCount) / numUtterances))
            
scpDict, utteranceIdList = createFormatDictionaryAndList(SCPFILE_TO_EXTRACT)
writeMultiChannelScpFile(scpDict, utteranceIdList)
saveMultiChannelWavFiles(scpDict, utteranceIdList)

