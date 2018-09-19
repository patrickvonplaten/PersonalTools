#!/usr/bin/env python3 

from tensorflow.contrib.framework.python.framework import checkpoint_utils
import numpy as np
import os, os.path
import ipdb 

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
        modelNames = sorted(['.'.join(x.split('.',2)[:2]) for x in os.listdir(self.modelDir) if '.data-' in x])

        self.numEpochs = len(modelNames)
        if(self.initWeightForLayerPath):
            self.saveWeight('network.000', self.initWeightForLayerPath, self.layerNameOfInitWeightsToSave)        

        for modelName in modelNames:
            self.saveWeight(modelName, self.modelDir + '/' + modelName, self.layerNameOfWeightsToSave)
        
        print('...saving weights of layer ' + self.nameOfLayerPath + ' done!')

    def saveWeight(self, modelName, modelPath, layerNameOfWeightsToSave):
        print("Save layer: " + layerNameOfWeightsToSave + " for epoch model " + modelPath)
        layerArray = checkpoint_utils.load_variable(modelPath, layerNameOfWeightsToSave)
        np.save(os.path.join(self.saveWeightsPath, self.nameOfLayerPath + '_' + modelName),layerArray) 
