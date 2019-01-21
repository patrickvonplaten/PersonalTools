#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import ipdb 



class ReturnnLayerPlotter(object):

    def __init__(self,pathToAnalysisDir, pathToWeights, nameOfLayer, plottingConfigs, numEpochs):
        self.pathToAnalysisDir = pathToAnalysisDir
        self.pathToWeights = pathToWeights
        self.nameOfLayer = nameOfLayer
        self.nameOfLayerPath = nameOfLayer.replace('/','_')
        self.plottingConfigs = plottingConfigs
        self.numEpochs = numEpochs
        self.epochRangeToPlotPerColumn = [self.numEpochs if x == 'numEpochs' else int(x) for x in self.plottingConfigs['plotRange']]
        self.reverse = self.plottingConfigs['reverse']
        self.layerType = self.plottingConfigs['layerType']
        self.weights = self.loadWeights()
        self.layer = self.getLayer()
        self.plotter = Plotter(self.pathToAnalysisDir, self.plottingConfigs, self.layer, self.epochRangeToPlotPerColumn)

    def loadWeights(self):
        weights = []
        for i in self.epochRangeToPlotPerColumn:
            epochWeight = np.load(os.path.join(self.pathToWeights, self.nameOfLayerPath + '_epoch' + str(i) + '_MMF.npy'))
            if(self.reverse):
                epochWeight = np.flip(epochWeight, axis=1)
            weights.append(np.squeeze(epochWeight))
        return weights

    def getLayer(self):
        lenWeightsTensor = len(self.weights[0].shape)
        isLayerWeightComposedOf1Subarrays = lenWeightsTensor == 1
        isLayerWeightComposedOf2Subarrays = lenWeightsTensor == 2
        isLayerWeightComposedOf3Subarrays = lenWeightsTensor == 3
        wishedPlottings = self.plottingConfigs['typeOfPlotting']
        isPlottingDomainLog = self.plottingConfigs['log']

        if(isLayerWeightComposedOf1Subarrays):
            return FeedForwardLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings, isPlottingDomainLog)
        elif(isLayerWeightComposedOf2Subarrays):
            if(self.layerType == 'conv'):
                return Conv1DLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings, isPlottingDomainLog) 
            elif(self.layerType == 'feed'):
                return FeedForwardLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings, isPlottingDomainLog)

        elif(isLayerWeightComposedOf3Subarrays):
            return Conv2DLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings) 

    def run(self):
        self.plotter.plot()
        print('...plotting graphs for ' + str(self.nameOfLayer) + ' done!')


class Layer(object):
    """This is an abstract class"""

    def __init__(self, weights, name, namePath, isPlottingDomainLog):
        self.isPlottingDomainLog = isPlottingDomainLog
        self.name = name
        self.namePath = namePath
        self.weights = weights
        self.numEpochs = len(self.weights)
        self.shape = self.weights[0].shape
        self.allowedPlottings = []
        self.plottingsToDo = []
        self.layerType = None
        self.dimInputIdx = 0
        self.dimInput = 1
    
    def setDomain(self, domain):
        self.domain = domain

    def setDimInputIdx(self, idx):
        self.dimInputIdx = idx
    
    def getPlotable3DWeights(self):
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


class ProcessedWeights(object):

    def __init__(self, weights, filterSize, isPlottingDomainLog, dimInput, timeFreqRatio=2): 
        self.dimInput = dimInput
        self.weights = weights
        self.filterSize = filterSize 
        self.timeFreqRatio = timeFreqRatio
        self.permutation = None 
        self.isPlottingDomainLog = isPlottingDomainLog
        self.timeAxisTime = np.arange(self.filterSize)
        self.timeAxisFreq = np.arange(int(self.filterSize/2))
        self.plotableWeights = self.create_plotable_weights(weights)
        self.plotableWeightsFreq, self.plotableWeightsFreqSorted = self.getFrequencyDomain()

    def create_plotable_weights(self, weights):
        self.plotableWeights = []
        for i in range(self.dimInput):
            self.plotableWeights.append([x for x in weights])
        return self.plotableWeights

    def getPlotable2DWeights(self, domain):
        if domain == 'time':
            return self.transformToHeatPlotableWeights(self.plotableWeights, 1)
        elif domain == 'freq':
            return self.transformToHeatPlotableWeights(self.plotableWeightsFreq, self.timeFreqRatio)

    def getPlotableSorted2DWeights(self):
        return self.transformToHeatPlotableWeights(self.plotableWeightsFreqSorted, self.timeFreqRatio)

    def transformToHeatPlotableWeights(self, listOfWeights, filterSizeRatio):
        timeOrFreqAxisDim = self.filterSize/filterSizeRatio
        timeOrFreqChannelDim = self.dimInput * timeOrFreqAxisDim 
        numChannels = len(listOfWeights)
        numEpochToPlot = len(listOfWeights[0])

        transformedList = []
        for j in range(numEpochToPlot):
            l = []
            for i in range(numChannels):
                l.append(np.swapaxes(listOfWeights[i][j],0,1))
            transformedList.append(l)

        return transformedList

    def getPlotable1DWeights(self, domain, dimInputIdx):
        if domain == 'time':
            return self.plotableWeights[dimInputIdx], self.timeAxisTime
        elif domain == 'freq':
            return self.plotableWeightsFreq[dimInputIdx], self.timeAxisFreq

    def fourierTransform(self, plotableWeightsSingleDim, sortFn):
            l = []
            for weightsEpoch in plotableWeightsSingleDim:
                layerWeightsFreqTransposed = np.fft.fft(weightsEpoch).T
                layerWeightsFreqTransposed = layerWeightsFreqTransposed[:int(len(layerWeightsFreqTransposed)/2)]
                layerWeightsFreqTransposedAbsolute = np.absolute(layerWeightsFreqTransposed.T)
                if(self.isPlottingDomainLog):
                    layerWeightsFreqTransposedAbsolute = np.log(layerWeightsFreqTransposedAbsolute)
                l.append(sortFn(layerWeightsFreqTransposedAbsolute))
            return l

    def getFrequencyDomain(self):
        timeAxis = np.arange(self.filterSize/self.timeFreqRatio)
        plotableWeightsFreq = [self.fourierTransform(x, self.noSortFreq) for x in self.plotableWeights]
        plotableWeightsFreqSorted = [self.fourierTransform(x, self.sortFreq) for x in self.plotableWeights]
        return plotableWeightsFreq, plotableWeightsFreqSorted

    def noSortFreq(self, x):
        return x

    def sortFreq(self, x):
        if(not isinstance(self.permutation,np.ndarray)):
            self.permutation = x.argmax(axis=1).argsort()
        return x[self.permutation] 


class FeedForwardLayer(Layer):
    
    def __init__(self, weights, name, namePath, wishedPlottings, isPlottingDomainLog):
        super(FeedForwardLayer, self).__init__(weights, name, namePath, isPlottingDomainLog)
        self.filterSize = self.shape[1]
        self.numFilters = self.shape[0]
        assert weights[0].shape[0] == self.numFilters, "dim not correct"
        assert weights[0].shape[1] == self.filterSize, "dim not correct"
        self.addToAllowedPlottings(['2DWeightsHeat'])
        self.setLayerType(type(self).__name__)
        self.createPlottingsToDo(wishedPlottings)
        self.processedWeights = ProcessedWeights(self.weights, self.filterSize, isPlottingDomainLog, self.dimInput)
        self.channelUsedForPermutation = 0 

    def getPlotable1DWeights(self):
        pass  

    def getPlotable2DWeights(self, domain):
        return self.processedWeights.getPlotable2DWeights(domain)

    def getPlotableSorted2DWeights(self):
        return self.processedWeights.getPlotableSorted2DWeights()

class Conv1DLayer(Layer):

    def __init__(self, weights, name, namePath, wishedPlottings, isPlottingDomainLog):
        super(Conv1DLayer, self).__init__(weights, name, namePath, isPlottingDomainLog)
        self.filterSize = self.shape[1]
        self.numFilters = self.shape[0]
        assert weights[0].shape[0] == self.numFilters, "dim not correct"
        assert weights[0].shape[1] == self.filterSize, "dim not correct"
        self.addToAllowedPlottings(['1DWeightsSimpleAll','1DWeightsSimpleDetail','2DWeightsHeat'])
        self.setLayerType(type(self).__name__)
        self.createPlottingsToDo(wishedPlottings)
        self.processedWeights = ProcessedWeights(self.weights, self.filterSize, isPlottingDomainLog, self.dimInput)
        self.channelUsedForPermutation = 0 

    def getPlotable1DWeights(self):
        pass  

    def getPlotable2DWeights(self, domain):
        return self.processedWeights.getPlotable2DWeights(domain)

    def getPlotableSorted2DWeights(self):
        return self.processedWeights.getPlotableSorted2DWeights()

class Conv2DLayer(Layer):
    
    def __init__(self, weights, name, namePath, wishedPlottings):
        super(Conv2DLayer, self).__init__(weights, name, namePath)
#        needs check!
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

    def __init__(self, pathToAnalysisDir, plottingConfigs, layer, epochRangeToPlotPerColumn):
        self.layer = layer
        self.pathToAnalysisDir = pathToAnalysisDir
        self.plottingConfigs = plottingConfigs
        self.colors = self.plottingConfigs['colors']
        self.epochRangeToPlotPerColumn = epochRangeToPlotPerColumn
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
                if('1DWeightsSimpleAll' in self.layer.plottingsToDo):
                    self.plot1DSimpleWeightsAll()
                if('1DWeightsSimpleDetail' in self.layer.plottingsToDo):
                    self.plot1DSimpleWeightsDetail()
            if('2DWeightsHeat' in self.layer.plottingsToDo):
                if(self.layer.domain == 'time'):
                    self.plot2DHeatWeights('unsorted')
                elif(self.layer.domain == 'freq'):
                    self.plot2DHeatWeights('sorted')
                    self.plot2DHeatWeights('unsorted')
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
        plt.savefig(self.pathToAnalysisDir + '/' + self.layer.namePath + '_lastEpoch_' + self.layer.domain + '_' + str(self.layer.dimInputIdx) + '_all')

    def plot1DSimpleWeightsDetail(self):
        assert 'plotRange' in self.plottingConfigs and len(self.plottingConfigs['plotRange']) > 1, 'plotting range needs to be defined and bigger than 1 - not yet supported' 
        assert 'samplesPerRow' in self.plottingConfigs, 'Needs to give the attribute samplesPerRow'
        assert type(self.layer).__name__ == 'Conv1DLayer', 'Currently only supported for Conv1DLayer - but could be changed by defining numFilters in other layers'
        assert self.layer.numFilters % self.samplesPerRow == 0, 'numFilters should be a multiple of samples per row'
        
        plotableWeights, timeArray = self.layer.getPlotable1DWeights()
        simple1DPlotDir = self.pathToAnalysisDir + '/' + '1DSimplePlots' + '_' + self.layer.domain+ '_' + str(self.layer.dimInputIdx) + 'dimIdx'

        if not os.path.exists(simple1DPlotDir):
            os.makedirs(simple1DPlotDir)

        for plotIdx in range(int(self.layer.numFilters/self.samplesPerRow)):
            fig, axs = plt.subplots(self.samplesPerRow, len(self.epochRangeToPlotPerColumn), figsize=self.figSize) 
            
            for epochRangeIdx, plotableWeight in enumerate(plotableWeights):
                for i in range(self.samplesPerRow):
                    filterNum = self.samplesPerRow*plotIdx + i
                    axs[i][epochRangeIdx].plot(timeArray, plotableWeight[filterNum],self.colors[i])
                    axs[i][epochRangeIdx].grid()
                    axs[i][epochRangeIdx].set_ylabel('filter.' + "%02d" % (filterNum+1,) + '_epoch.'+ "%03d" % (self.epochRangeToPlotPerColumn[epochRangeIdx],) + '_weights')
                    axs[i][epochRangeIdx].set_xlabel(self.layer.domain) 

            nameToSave = self.layer.namePath + '_' + self.layer.domain + '_' + str(self.layer.dimInputIdx) + '_' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow)

            plt.suptitle(self.title + '_' + self.layer.domain + '_' + str(self.layer.dimInputIdx) + 'dimIdx' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow), fontsize=self.titleFontSize, y=self.titleYPosition)
            plt.savefig(simple1DPlotDir + '/' + nameToSave)
            print('...saving ' + nameToSave)
    
    def plot2DHeatWeights(self, mode):
        assert mode in ['sorted','unsorted'], "mode has to be sorted or unsorted"
        
        fig, axs = plt.subplots(self.layer.dimInput, len(self.epochRangeToPlotPerColumn), figsize=self.figSize, sharex=True, sharey=True)

        if(self.layer.dimInput == 1 and not isinstance(axs, np.ndarray)):
            axs = [[axs]]
        elif(self.layer.dimInput == 1):
            axs = [axs]
        elif(len(axs.shape) < 2):
            axs = [[ax] for ax in axs]

        if(mode == 'sorted'):
            plotableWeights = self.layer.getPlotableSorted2DWeights()
            mode = 'sorted_by_channel_' + str(self.layer.channelUsedForPermutation)
        elif(mode == 'unsorted'):
            plotableWeights = self.layer.getPlotable2DWeights(self.layer.domain)

        for epochRangeIdx, plotableWeight in enumerate(plotableWeights):
            for dimInputIdx, plotableWeightPerDim in enumerate(plotableWeight): 
                im = axs[dimInputIdx][epochRangeIdx].imshow(plotableWeightPerDim, origin='lower', aspect='auto', cmap=self.plottingConfigs['cmap'])
                axs[dimInputIdx][epochRangeIdx].set_ylabel(self.layer.domain + '_for_channel_' + str(dimInputIdx))
                axs[dimInputIdx][epochRangeIdx].set_xlabel('filterIdx_' + mode + '_for epoch' + '_' + '%03d' % (self.epochRangeToPlotPerColumn[epochRangeIdx],))

        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(right=0.8)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        numDomain = '_log_applied' if(self.layer.isPlottingDomainLog and self.layer.domain == 'freq') else ''
        plt.suptitle(self.title + '_' + self.layer.domain + '_for_epoch_' + '_'.join(str(x) for x in self.epochRangeToPlotPerColumn) + numDomain, fontsize=self.titleFontSize, y=self.titleYPosition)
        plt.savefig(self.pathToAnalysisDir + '/' + self.layer.namePath + '_heat_map_' + self.layer.domain + '_' + mode + numDomain)
        
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

            nameToSave = self.layer.namePath + '_' + self.layer.domain + '_' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow)

            plt.suptitle(self.title + '_' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow), fontsize=self.titleFontSize, y=self.titleYPosition)
            plt.savefig(simple1DPlotDir + '/' + nameToSave)
        
