#!/usr/bin/env python3 

import numpy as np
import os, os.path
import ipdb 
import pyhtk

class LayerWeightSaver(object):

    def __init__(self, modelDir, saveWeightsPath, nameOfLayer, initWeightForLayerPath=None, layerNameForInitWeight=None):
        self.modelDir = modelDir
        self.saveWeightsPath = saveWeightsPath
        self.layerNameOfWeightsToSave = nameOfLayer
        self.nameOfLayerPath = nameOfLayer.replace('/','_')
        self.numEpochs = None
        self.initWeightForLayerPath = initWeightForLayerPath
        if(self.initWeightForLayerPath):
            self.layerNameOfInitWeightsToSave = layerNameForInitWeight 
        self.saveWeights()

    def saveWeights(self):
        modelNames = [ x + '/MMF' for x in os.listdir(self.modelDir) if 'epoch' in x ]
        self.numEpochs = len(modelNames)
        for modelName in modelNames:
            self.saveWeight('_'.join(modelName.split('/')), self.modelDir + '/' + modelName, self.layerNameOfWeightsToSave)
        
        print('...saving weights of layer ' + self.nameOfLayerPath + ' done!')

    def saveWeight(self, modelName, modelPath, layerNameOfWeightsToSave):
        print("Save layer: " + layerNameOfWeightsToSave + " for epoch model " + modelPath)
        hmmSet = pyhtk.HTKModelReader(modelPath, '').getHiddenMarkovModelSet()
        layerArray = hmmSet.getNMatrixTable()[layerNameOfWeightsToSave].getValuesAsNumPyArray()
        np.save(os.path.join(self.saveWeightsPath, self.nameOfLayerPath + '_' + modelName),layerArray) 
