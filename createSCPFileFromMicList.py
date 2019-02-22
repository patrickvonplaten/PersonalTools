#!/usr/bin/env python3
import sys
import os
import numpy as np
import ipdb

SCPFILE_TO_READ=sys.argv[1]
MICLISTFILE_TO_READ=sys.argv[2]
SCPFILE_TO_WRITE=sys.argv[3]
ABS_SINGLECHANNEL_FOLDER='/home/dawna/chime/import/data/CHiME4/data/audio/16kHz/'
RELATIV_FOLDER_TO_COPY = 2
CHANNEL_ID_TO_REPLACE = '05'

def transformPathToWavFileToFormatScpFileName(scpPathToWavFile):
    utterancePath = scpPathToWavFile.split('/')[-1]
    utteranceId = utterancePath.split('.')[0]
    utteranceInterval = utterancePath.split('[')[-1]
    return utteranceId + '.CH{}.wav[' + utteranceInterval

def getRelativPath(scpPathToWavFile):
    return '/'.join(scpPathToWavFile.split('/')[-1-RELATIV_FOLDER_TO_COPY:-1])

def transformToFormatBabelUtteranceName(babelUtteranceName, channelIdToReplace):
    babelUtteranceNameItems = babelUtteranceName.split('-' + channelIdToReplace + '_')
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

def writeScpFile(scpDict, micDict, uttList, scpFilePath=SCPFILE_TO_WRITE):
    with open(scpFilePath, 'w+') as scpFile:
        for key in uttList:
            channelToUse = micDict[key]
            lineToWrite = key.format(CHANNEL_ID_TO_REPLACE) + '=' + os.path.join(ABS_SINGLECHANNEL_FOLDER, scpDict[key].format(channelToUse)) + '\n'
            scpFile.write(lineToWrite)

def createMicDict(micListFilePath):
    micDict = {}
    with open(micListFilePath, 'r') as micFile:
        lines = micFile.readlines()
        for line in lines:
            items = line.strip().split('=')
            micDict[transformToFormatBabelUtteranceName(items[0], CHANNEL_ID_TO_REPLACE) + '.wav']=int(items[1])
    return micDict

scpDict, uttList = createFormatDictionaryAndList(SCPFILE_TO_READ)
micDict = createMicDict(MICLISTFILE_TO_READ)
writeScpFile(scpDict, micDict, uttList)
