#!/usr/bin/env python3 

from tensorflow.contrib.framework.python.framework import checkpoint_utils
import numpy as np
import os, os.path

class LayerWeightSaver(object):

    def __init__(self, modelDir, saveWeightsPath, nameOfLayer, parameterTypeToSave):
        self.modelDir = modelDir
        self.saveWeightsPath = saveWeightsPath
        self.nameOfLayer = nameOfLayer
        self.nameOfLayerParametersToSave = nameOfLayer + '/' + parameterTypeToSave
        self.numEpochs = None
        self.saveWeights()

    def saveWeights(self):
        modelNames = sorted(['.'.join(x.split('.',2)[:2]) for x in os.listdir(self.modelDir) if '.data-' in x])

        self.numEpochs = len(modelNames)
        for modelName in modelNames:
            self.saveWeight(modelName, self.modelDir + '/' + modelName)
        
        print('...saving weights of layer ' + self.nameOfLayer + ' done!')

    def saveWeight(self, modelName, modelPath):
        print("Save layer: " + self.nameOfLayerParametersToSave + " for epoch model " + modelPath)
        layerArray = checkpoint_utils.load_variable(modelPath, self.nameOfLayerParametersToSave)
        np.save(os.path.join(self.saveWeightsPath, self.nameOfLayer + '_' + modelName),layerArray) 
