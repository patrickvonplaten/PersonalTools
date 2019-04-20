#!/usr/bin/env python3 

import numpy as np
import os, os.path
import ipdb 
import pyhtk

class LayerWeightSaver(object):

    def __init__(self, modelDir, saveWeightsPath, nameOfLayer, pathToInitWeight, channelNums):
        self.modelDir = modelDir
        self.saveWeightsPath = saveWeightsPath
        self.layerNameOfWeightsToSave = nameOfLayer
        self.nameOfLayerPath = nameOfLayer.replace('/','_')
        self.numEpochs = None
        self.pathToInitWeight = pathToInitWeight
        self.channelNums = channelNums
        self.saveWeights()

    def saveWeights(self):
        modelNames = [ x + '/MMF' for x in os.listdir(self.modelDir) if 'epoch' in x ]
        self.numEpochs = len(modelNames)
        if(self.pathToInitWeight):
            self.saveWeight('epoch0_MMF', self.pathToInitWeight)
        for modelName in modelNames:
            print('_'.join(modelName.split('/')), self.modelDir + '/' + modelName)
            self.saveWeight('_'.join(modelName.split('/')), self.modelDir + '/' + modelName,self.channelNums)
        print('...saving weights of layer ' + self.nameOfLayerPath + ' done!')

    def saveWeight(self, modelName, modelPath, channelNums):
        layerArray = []
        hmmSet = pyhtk.HTKModelReader(modelPath, '').getHiddenMarkovModelSet()
        print("Save layer: " + self.layerNameOfWeightsToSave + " for epoch model " + modelPath)
        for i in range(channelNums):
            layerNameExtension = 'Channel{}'.format(i+1) if channelNums > 1 else ''
            layerName = '_'.join(self.layerNameOfWeightsToSave.split('_')[:-1]) + layerNameExtension + '_' + self.layerNameOfWeightsToSave.split('_')[-1]
            nMatrixTable = hmmSet.getNMatrixTable()[layerName]
            layerArray.append(nMatrixTable.getValuesAsNumPyArray())
        layerArrayToSave = np.asarray(layerArray)
        if(len(layerArrayToSave.shape) > 2):
            layerArrayToSave = np.moveaxis(layerArrayToSave, 0, -1)
        np.save(os.path.join(self.saveWeightsPath, self.nameOfLayerPath + '_' + modelName),layerArrayToSave) 
