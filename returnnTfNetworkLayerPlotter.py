#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os
import sys

class ReturnnLayerPlotter(object):

    def __init__(self,pathToAnalysisDir, pathToWeights, nameOfLayer, plottingConfigs, numEpochs):
        self.pathToAnalysisDir = pathToAnalysisDir
        self.pathToWeights = pathToWeights
        self.nameOfLayer = nameOfLayer
        self.plottingConfigs = plottingConfigs
        self.numEpochs = numEpochs

        self.weights = self.loadWeights()
        self.layer = self.getLayer()
        self.plotter = Plotter(self.pathToAnalysisDir, self.plottingConfigs, self.layer)

    def loadWeights(self):
        weights = []
        for i in range(1,self.numEpochs+1):
            weights.append(np.load(os.path.join(self.pathToWeights, self.nameOfLayer + '_network.' + "%03d" %             (i,) + '.npy'))) 
        return weights

    def getLayer(self):
        lenWeightsTensor = len(self.weights[0].shape)
        isLayerWeightComposedOf2Subarrays = lenWeightsTensor == 2
        isLayerWeightComposedOf3Subarrays = lenWeightsTensor == 3 
        isLayerWeightComposedOf4Subarrays = lenWeightsTensor == 4
        wishedPlottings = self.plottingConfigs['typeOfPlotting']

        if(isLayerWeightComposedOf2Subarrays):
            return FeedForwardLayer(self.weights, self.nameOfLayer, wishedPlottings) 
        elif(isLayerWeightComposedOf3Subarrays):
            return Conv1DLayer(self.weights, self.nameOfLayer, wishedPlottings) 
        elif(isLayerWeightComposedOf4Subarrays):
            return Conv2DLayer(self.weights, self.nameOfLayer, wishedPlottings) 

    def run(self):
        self.plotter.plot()
        print('...plotting graphs for ' + str(self.nameOfLayer) + ' done!')

class Layer(object):
    """This is an abstract class"""

    def __init__(self, weights, name):
        self.weights = weights
        self.name = name
        self.numEpochs = len(self.weights)
        self.shape = self.weights[0].shape
        self.allowedPlottings = []
        self.plottingsToDo = []
        self.layerType = None
        self.domain = None 
        self.dimInputIdx = 0
        self.dimInput = 1  
    
    def setDomain(self, domain):
        self.domain = domain

    def setDimInputIdx(self, idx):
        self.dimInputIdx = idx
    
    def getPlotable3DWeights(self):
        raise NotImplementedError

    def getPlotable2DWeights(self):
        raise NotImplementedError

    def getPlotable1DWeights(self):
        raise NotImplementedError
    
    def transformWeightsToFrequencyDomain(self):
        raise NotImplementedError

    def addToAllowedPlottings(self, plottings):
        for plotting in plottings:
            self.allowedPlottings.append(plotting)

    def createPlottingsToDo(self, wishedPlottings):
        for plotting in wishedPlottings:
            if(plotting in self.allowedPlottings):
                self.plottingsToDo.append(plotting)
            else:
                print('Plotting ' + plotting + ' is not allowed for ' + self.layerType) 
   
    def setLayerType(self, layerType):
        self.layerType = layerType 
    
class FeedForwardLayer(Layer):
    
    def __init__(self, weights, name, wishedPlottings):
        super(FeedForwardLayer, self).__init__(weights, name)
        self.plotableWeightsTime = [x[1] for x in self.weights]
        self.bias = [x[0] for x in self.weights]
        self.addToAllowedPlottings(['2DWeightsHeat'])
        self.setLayerType(type(self).__name__)
        self.createPlottingsToDo(wishedPlottings)
        
    def getPlotable2DWeights(self):
        assert self.dimInputIdx == 0, "for feed forward layer there is always only one dim"
        return self.plotableWeightsTime

    def getPlotable1DWeights(self):
        pass  

class Conv1DLayer(Layer):

    def __init__(self, weights, name, wishedPlottings):
        super(Conv1DLayer, self).__init__(weights, name)
        self.filterSize = self.shape[0]
        self.numFilters = self.shape[-1]
        self.dimInput = self.shape[1]
        self.weightsReshaped = [np.moveaxis(x,0, -1) for x in self.weights] 
        assert self.weightsReshaped[0].shape[0] == self.dimInput, "dim not correct"
        assert self.weightsReshaped[0].shape[1] == self.numFilters, "dim not correct"
        assert self.weightsReshaped[0].shape[2] == self.filterSize, "dim not correct"
        self.plotableWeightsTime = []
    
        for i in range(self.dimInput):
            self.plotableWeightsTime.append([x[i] for x in self.weightsReshaped])
            
        self.plotableWeightsFreq, self.plotableWeightsFreqSorted = self.getFrequencyDomain()
        self.timeAxisTime = np.arange(self.filterSize)
        self.timeAxisFreq = np.arange(int(self.filterSize/2))
        self.name = name
        self.addToAllowedPlottings(['1DWeightsSimpleAll','1DWeightsSimpleDetail','2DWeightsHeat'])
        self.setLayerType(type(self).__name__)
        self.createPlottingsToDo(wishedPlottings)

    def getPlotable2DWeights(self):
        if self.domain == 'time':
            return self.plotableWeightsTime[self.dimInputIdx]
        elif self.domain == 'freq':
            return self.plotableWeightsFreq[self.dimInputIdx]
    
    def getPlotableSorted2DWeights(self):
        return self.plotableWeightsFreqSorted[self.dimInputIdx]

    def getPlotable1DWeights(self):
        if self.domain == 'time':
            return self.plotableWeightsTime[self.dimInputIdx], self.timeAxisTime
        elif self.domain == 'freq':
            return self.plotableWeightsFreq[self.dimInputIdx], self.timeAxisFreq

    def getFrequencyDomain(self):
        timeAxis = np.arange(self.filterSize/2)
        plotableWeightsFreq = [self.fourierTransform(x, self.noSortFreq) for x in self.plotableWeightsTime]
        plotableWeightsFreqSorted = [self.fourierTransform(x, self.sortFreq) for x in self.plotableWeightsTime]
        return plotableWeightsFreq, plotableWeightsFreqSorted

    def fourierTransform(self, plotableWeightsTimeSingleDim, sortFn):
            l = []
            for weightsEpoch in plotableWeightsTimeSingleDim:
                layerWeightsFreqTransposed = np.fft.fft(weightsEpoch).T
                layerWeightsFreqTransposed = layerWeightsFreqTransposed[:int(len(layerWeightsFreqTransposed)/2)]
                layerWeightsFreqTransposedAbsolute = np.absolute(layerWeightsFreqTransposed.T)
                l.append(sortFn(layerWeightsFreqTransposedAbsolute))
            return l

    def noSortFreq(self, x):
        return x

    def sortFreq(self, x):
        return x[x.argmax(axis=1).argsort()]

class Conv2DLayer(Layer):
    
    def __init__(self, weights, name, wishedPlottings):
        super(Conv2DLayer, self).__init__(weights, name)
        filterSizeDim1 = self.shape[0]
        filterSizeDim2 = self.shape[1]
        numFilters = self.shape[-1]
        self.plotable3DWeights = [x.reshape((filterSizeDim1, filterSizeDim2, numFilters)).T for x in self.weights]
        self.name = name
        self.addToAllowedPlottings(['3DWeightsHeat'])
        self.setLayerType(type(self).__name__)
        self.createPlottingsToDo(wishedPlottings)

    def getPlotable3DWeights(self):
        return self.plotable3DWeights
        
class Plotter(object):

    def __init__(self, pathToAnalysisDir, plottingConfigs, layer):
        self.layer = layer
        self.pathToAnalysisDir = pathToAnalysisDir
        self.plottingConfigs = plottingConfigs
        self.colors = self.plottingConfigs['colors']
        self.epochRangeToPlotPerColumn = [self.layer.numEpochs if x == 'numEpochs' else int(x) for x in self.plottingConfigs['plotRange']]
        self.samplesPerRow = self.plottingConfigs['samplesPerRow']
        self.samplesPerColumn = self.plottingConfigs['samplesPerColumn']
        self.figSize = self.plottingConfigs['figSize']
        self.title = self.plottingConfigs['title']
        self.titleFontSize = 30 
        self.titleYPosition = 0.93
            
    def plot(self):
        for domain in self.plottingConfigs['domainType']:
            self.layer.setDomain(domain)
            for inputDimIdx in range(self.layer.dimInput):
                self.layer.setDimInputIdx(inputDimIdx)
                print('...saving plots for ' + self.layer.name + ' for dimIdx ' + str(inputDimIdx) + ' in ' + self.layer.domain)
                if('1DWeightsSimpleAll' in self.layer.plottingsToDo):
                    self.plot1DSimpleWeightsAll()
                if('1DWeightsSimpleDetail' in self.layer.plottingsToDo):
                    self.plot1DSimpleWeightsDetail()
                if('2DWeightsHeat' in self.layer.plottingsToDo):
                    if(self.layer.domain == 'time'):
                        self.plot2DHeatWeights()
                    elif(self.layer.domain == 'freq'):
                        self.plot2DHeatFreqWeights()
                if('3DWeightsHeat' in self.layer.plottingsToDo):
                     self.plot3DHeatWeights()
         
    def plot1DSimpleWeightsAll(self):
        assert 'samplesPerRow' in self.plottingConfigs, 'Needs to give the attribute samplesPerRow'
        assert 'samplesPerColumn' in self.plottingConfigs, 'Needs to give the attribute samplesPerColumn'

        fig, axs = plt.subplots(self.samplesPerColumn, self.samplesPerRow, figsize=self.plottingConfigs['figSize'])
        plotableWeights, timeArray = self.layer.getPlotable1DWeights()
        
        for i in range(self.samplesPerColumn):
            for j in range(self.samplesPerRow):
                filterNum = self.samplesPerRow*i+j
                axs[i][j].plot(timeArray,plotableWeights[-1][filterNum],self.colors[j])
                axs[i][j].grid()
                axs[i][j].set_ylabel('filter.' + "%02d" % (filterNum+1,))
                
        plt.suptitle(self.title + '_' + self.layer.domain + '_' + str(self.layer.dimInputIdx) + 'dimIdx', fontsize=self.titleFontSize, y=self.titleYPosition)
        plt.savefig(self.pathToAnalysisDir + '/' + self.layer.name + '_lastEpoch_' + self.layer.domain + '_' + str(self.layer.dimInputIdx) + '_all')

    def plot1DSimpleWeightsDetail(self):
        assert 'plotRange' in self.plottingConfigs and len(self.plottingConfigs['plotRange']) > 1, 'plotting range needs to be defined and bigger than 1'
        assert 'samplesPerRow' in self.plottingConfigs, 'Needs to give the attribute samplesPerRow'
        assert type(self.layer).__name__ == 'Conv1DLayer', 'Currently only supported for Conv1DLayer - but could be changed by defining numFilters in other layers'
        assert self.layer.numFilters % self.samplesPerRow == 0, 'numFilters should be a multiple of samples per row'
        
        plotableWeights, timeArray = self.layer.getPlotable1DWeights()
        simple1DPlotDir = self.pathToAnalysisDir + '/' + '1DSimplePlots' + '_' + self.layer.domain+ '_' + str(self.layer.dimInputIdx) + 'dimIdx'

        if not os.path.exists(simple1DPlotDir):
            os.makedirs(simple1DPlotDir)

        for plotIdx in range(int(self.layer.numFilters/self.samplesPerRow)):
            fig, axs = plt.subplots(self.samplesPerRow, len(self.epochRangeToPlotPerColumn), figsize=self.figSize) 
            
            for epochRangeIdx, epoch in enumerate(self.epochRangeToPlotPerColumn):
                for i in range(self.samplesPerRow):
                    filterNum = self.samplesPerRow*plotIdx + i
                    axs[i][epochRangeIdx].plot(timeArray, plotableWeights[epoch-1][filterNum],self.colors[i])
                    axs[i][epochRangeIdx].grid()
                    axs[i][epochRangeIdx].set_ylabel('filter.' + "%02d" % (filterNum+1,) + '_epoch.'+ "%03d" % (epoch,) + '_weights')
                    axs[i][epochRangeIdx].set_xlabel(self.layer.domain) 

            nameToSave = self.layer.name + '_' + self.layer.domain + '_' + str(self.layer.dimInputIdx) + '_' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow)

            plt.suptitle(self.title + '_' + self.layer.domain + '_' + str(self.layer.dimInputIdx) + 'dimIdx' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow), fontsize=self.titleFontSize, y=self.titleYPosition)
            plt.savefig(simple1DPlotDir + '/' + nameToSave)
            print('...saving ' + nameToSave)
    
    def plot2DHeatWeights(self):
        assert 'plotRange' in self.plottingConfigs and len(self.plottingConfigs['plotRange']) > 1, 'plotting range needs to be defined and bigger than 1'

        fig, axs = plt.subplots(len(self.epochRangeToPlotPerColumn),1, figsize=self.figSize)
        axs = axs.ravel()
        plotableWeights = self.layer.getPlotable2DWeights()

        for epochRangeIdx, epoch in enumerate(self.epochRangeToPlotPerColumn):
            im = axs[epochRangeIdx].imshow(plotableWeights[epoch-1], origin='lower', aspect='auto')
            axs[epochRangeIdx].set_title('network. ' + '%03d' % (epoch,))
            axs[epochRangeIdx].set_ylabel('filters')
            axs[epochRangeIdx].set_xlabel(self.layer.domain) 

        fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im, cax=cbar_ax)

        plt.suptitle(self.title + '_' + self.layer.domain + '_' + str(self.layer.dimInputIdx) + 'dimIdx', fontsize=self.titleFontSize, y=self.titleYPosition)
	plt.savefig(self.pathToAnalysisDir + '/' + self.layer.name + '_heat_map_' + self.layer.domain + '_' + str(self.layer.dimInputIdx))
    
    def plot2DHeatFreqWeights(self):
        assert 'plotRange' in self.plottingConfigs and len(self.plottingConfigs['plotRange']) > 1, 'plotting range needs to be defined and bigger than 1'

        fig, axs = plt.subplots(len(self.epochRangeToPlotPerColumn),2, figsize=self.figSize)

        plotableWeightsList = [self.layer.getPlotable2DWeights(), self.layer.getPlotableSorted2DWeights()]
        sortingStr = ['unsorted','sorted']
        
        for idx, plotableWeights in enumerate(plotableWeightsList):
            for epochRangeIdx, epoch in enumerate(self.epochRangeToPlotPerColumn):
                im = axs[epochRangeIdx][idx].imshow(plotableWeights[epoch-1], origin='lower', aspect='auto')
                axs[epochRangeIdx][idx].set_title('network._' + sortingStr[idx] + '_' + '%03d' % (epoch,))
                axs[epochRangeIdx][idx].set_ylabel('filters')
                axs[epochRangeIdx][idx].set_xlabel(self.layer.domain)

        fig.subplots_adjust(right=0.8)
	cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
	fig.colorbar(im, cax=cbar_ax)

        plt.suptitle(self.title + '_' + self.layer.domain + '_' + str(self.layer.dimInputIdx) + 'dimIdx', fontsize=self.titleFontSize, y=self.titleYPosition)
	plt.savefig(self.pathToAnalysisDir + '/' + self.layer.name + '_heat_map_' + self.layer.domain + '_' + str(self.layer.dimInputIdx))
        
    def plot3DHeatWeights(self):
        plotableWeights = self.layer.getPlotable3DWeights
        heat3DWeightsDir = self.pathToAnalysisDir + '/' + '2DHeatPlotsForFilter' + '_' + self.layer.domain
        
        if not os.path.exists(heat3DWeightsDir):
            os.makedirs(heat3DWeightsDir)

#        TODO:Make it work!
        for plotIdx in range(int(self.layer.numFilters/self.samplesPerRow)):
            fig, axs = plt.subplots(self.samplesPerRow, len(self.epochRangeToPlotPerColumn), figsize=self.figSize) 
            
            for epochRangeIdx, epoch in enumerate(self.epochRangeToPlotPerColumn):
                for i in range(self.samplesPerRow):
                    filterNum = self.samplesPerRow*plotIdx + i
                    axs[i][epochRangeIdx].plot(timeArray, plotableWeights[epoch-1][filterNum],self.colors[i])
                    axs[i][epochRangeIdx].grid()
                    axs[i][epochRangeIdx].set_ylabel('filter.' + "%02d" % (filterNum+1,) + '_epoch.'+ "%03d" % (epoch,) + '_weights')
                    axs[i][epochRangeIdx].set_xlabel(self.layer.domain) 

            nameToSave = self.layer.name + '_' + self.layer.domain + '_' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow)

            plt.suptitle(self.title + '_' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow), fontsize=self.titleFontSize, y=self.titleYPosition)
            plt.savefig(simple1DPlotDir + '/' + nameToSave)
        
