#!/usr/bin/env python3 

import numpy as np
import os, os.path
import ipdb 
import pyhtk

class LayerWeightSaver(object):

    def __init__(self, modelDir, saveWeightsPath, nameOfLayer, pathToInitWeight=None):
        self.modelDir = modelDir
        self.saveWeightsPath = saveWeightsPath
        self.layerNameOfWeightsToSave = nameOfLayer
        self.nameOfLayerPath = nameOfLayer.replace('/','_')
        self.numEpochs = None
        self.pathToInitWeight = pathToInitWeight
        self.saveWeights()

    def saveWeights(self):
        modelNames = [ x + '/MMF' for x in os.listdir(self.modelDir) if 'epoch' in x ]
        self.numEpochs = len(modelNames)
        if(self.pathToInitWeight):
            self.saveWeight('epoch0_MMF', self.pathToInitWeight)
        for modelName in modelNames:
            self.saveWeight('_'.join(modelName.split('/')), self.modelDir + '/' + modelName)
        
        print('...saving weights of layer ' + self.nameOfLayerPath + ' done!')

    def saveWeight(self, modelName, modelPath):
        print("Save layer: " + self.layerNameOfWeightsToSave + " for epoch model " + modelPath)
        hmmSet = pyhtk.HTKModelReader(modelPath, '').getHiddenMarkovModelSet()
        layerArray = hmmSet.getNMatrixTable()[self.layerNameOfWeightsToSave].getValuesAsNumPyArray()
        np.save(os.path.join(self.saveWeightsPath, self.nameOfLayerPath + '_' + modelName),layerArray) 
