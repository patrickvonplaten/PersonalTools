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
        self.epochRangeToPlotPerColumn = [self.numEpochs + int(x) + 1 if int(x) < 0 else int(x) for x in self.plottingConfigs['plotRange']]
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
        doPaddedFourierTransform = self.plottingConfigs['pad']
        sampleRate = self.plottingConfigs['sampleRate']

        if(isLayerWeightComposedOf1Subarrays):
            return FeedForwardLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings, isPlottingDomainLog, doPaddedFourierTransform, sampleRate)
        elif(isLayerWeightComposedOf2Subarrays):
            if(self.layerType == 'conv'):
                return Conv1DLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings, isPlottingDomainLog, doPaddedFourierTransform, sampleRate) 
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
        self.domain = None
    
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

    def __init__(self, weights, filterSize, isPlottingDomainLog, dimInput, doPaddedFourierTransform, sampleRate, timeFreqRatio=2): 
        self.dimInput = dimInput
        self.weights = weights
        self.filterSize = filterSize 
        self.timeFreqRatio = timeFreqRatio
        self.permutation = None 
        self.isPlottingDomainLog = isPlottingDomainLog
        self.doPaddedFourierTransform = doPaddedFourierTransform
        self.sampleRate = sampleRate
        self.timeAxisTime = np.arange(self.filterSize)
        self.timeAxisFreq = np.arange(int(self.filterSize/2))
        self.plotableWeights = self.create_plotable_weights(weights)
        self.plotableWeightsFreq, self.plotableWeightsFreqSorted = self.getFrequencyDomain()

    def getPlotable2DWeights(self, domain):
        if domain == 'time':
            return self.transformToHeatPlotableWeights(self.plotableWeights, 1)
        elif domain == 'freq':
            return self.transformToHeatPlotableWeights(self.plotableWeightsFreq, self.timeFreqRatio)

    def getPlotable1DWeights(self, domain, dimInputIdx):
        if domain == 'time':
            return self.plotableWeights[dimInputIdx], self.timeAxisTime
        elif domain == 'freq':
            return self.plotableWeightsFreq[dimInputIdx], self.timeAxisFreq

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

    def create_plotable_weights(self, weights):
        self.plotableWeights = []
        for i in range(self.dimInput):
            self.plotableWeights.append([x for x in weights])
        return self.plotableWeights

    def getFrequencyDomain(self):
        timeAxis = np.arange(self.filterSize/self.timeFreqRatio)
        plotableWeightsFreq = [self.fourierTransform(x, self.noSortFreq) for x in self.plotableWeights]
        plotableWeightsFreqSorted = [self.fourierTransform(x, self.sortFreq) for x in self.plotableWeights]
        return plotableWeightsFreq, plotableWeightsFreqSorted

    def fourierTransform(self, plotableWeightsSingleDim, sortFn):
            l = []
            for weightsEpoch in plotableWeightsSingleDim:
                if(self.doPaddedFourierTransform):
                    shapePadded = (weightsEpoch.shape[0], self.sampleRate)
                    weightsEpochPadded = np.zeros(shapePadded)
                    weightsEpochPadded[:,:self.filterSize] = weightsEpoch
                    weightsEpoch = weightsEpochPadded

                layerWeightsFreqTransposed = np.fft.fft(weightsEpoch).T
                layerWeightsFreqTransposed = layerWeightsFreqTransposed[:int(len(layerWeightsFreqTransposed)/2)]
                layerWeightsFreqTransposedAbsolute = np.absolute(layerWeightsFreqTransposed.T)
                if(self.isPlottingDomainLog):
                    layerWeightsFreqTransposedAbsolute = np.log(layerWeightsFreqTransposedAbsolute)
                l.append(sortFn(layerWeightsFreqTransposedAbsolute))
            return l

    def noSortFreq(self, x):
        return x

    def sortFreq(self, x):
        if(not isinstance(self.permutation,np.ndarray)):
            self.permutation = x.argmax(axis=1).argsort()
        return x[self.permutation] 


class FeedForwardLayer(Layer):
    
    def __init__(self, weights, name, namePath, wishedPlottings, isPlottingDomainLog, doPaddedFourierTransform, sampleRate):
        super(FeedForwardLayer, self).__init__(weights, name, namePath, isPlottingDomainLog)
        self.filterSize = self.shape[1]
        self.numFilters = self.shape[0]
        assert weights[0].shape[0] == self.numFilters, "dim not correct"
        assert weights[0].shape[1] == self.filterSize, "dim not correct"
        self.addToAllowedPlottings(['1DWeightsSimpleAll','2DWeightsHeat'])
        self.setLayerType(type(self).__name__)
        self.createPlottingsToDo(wishedPlottings)
        self.processedWeights = ProcessedWeights(self.weights, self.filterSize, isPlottingDomainLog, self.dimInput, doPaddedFourierTransform, sampleRate)
        self.channelUsedForPermutation = 0 

    def getPlotable1DWeights(self):
        return self.processedWeights.getPlotable1DWeights(self.domain, self.dimInputIdx)

    def getPlotable2DWeights(self):
        return self.processedWeights.getPlotable2DWeights(self.domain)

    def getPlotableSorted2DWeights(self):
        return self.processedWeights.getPlotableSorted2DWeights()

class Conv1DLayer(Layer):

    def __init__(self, weights, name, namePath, wishedPlottings, isPlottingDomainLog, doPaddedFourierTransform, sampleRate):
        super(Conv1DLayer, self).__init__(weights, name, namePath, isPlottingDomainLog)
        self.filterSize = self.shape[1]
        self.numFilters = self.shape[0]
        assert weights[0].shape[0] == self.numFilters, "dim not correct"
        assert weights[0].shape[1] == self.filterSize, "dim not correct"
        self.addToAllowedPlottings(['1DWeightsSimpleAll','1DWeightsSimpleDetail','2DWeightsHeat'])
        self.setLayerType(type(self).__name__)
        self.createPlottingsToDo(wishedPlottings)
        self.processedWeights = ProcessedWeights(self.weights, self.filterSize, isPlottingDomainLog, self.dimInput, doPaddedFourierTransform, sampleRate)
        self.channelUsedForPermutation = 0 

    def getPlotable1DWeights(self):
        return self.processedWeights.getPlotable1DWeights(self.domain, self.dimInputIdx)

    def getPlotable2DWeights(self):
        return self.processedWeights.getPlotable2DWeights(self.domain)

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
        self.samplesPerColumn = int(self.plottingConfigs['samplesPerColumn']) if int(self.plottingConfigs['samplesPerColumn']) > 0 else None
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

        plotableWeights, timeArray = self.layer.getPlotable1DWeights()
        numFilters = plotableWeights[0].shape[0]
        if(not self.samplesPerColumn):
            self.samplesPerColumn = int(np.ceil(numFilters/float(self.samplesPerRow)))

        
        for epochRangeIdx, plotableWeight in enumerate(plotableWeights):
            fig, axs = plt.subplots(self.samplesPerColumn, self.samplesPerRow, figsize=self.plottingConfigs['figSize'])
            for i in range(self.samplesPerColumn):
                for j in range(self.samplesPerRow):
                    filterNum = self.samplesPerRow*i+j
                    if(filterNum < numFilters):
                        self.plot1DGraph(axs, i, j, timeArray, plotableWeight[filterNum], j)
                        self.setGraphYAxisLable(axs, i, j, 'filter.' + "%02d" % (filterNum+1,))
                
            plotId = '_epoch' + str(self.epochRangeToPlotPerColumn[epochRangeIdx]) + '_' + self.layer.domain + '_dimIdx' + str(self.layer.dimInputIdx) + '_all'
            self.savePlot(plt, plotId)
            self.setPlotTitle(plt, plotId)

    def savePlot(self, plot, plotId):
        plot.savefig(self.pathToAnalysisDir + '/' + self.layer.namePath + plotId)

    def setPlotTitle(self, plot, plotId):
        plt.suptitle(self.title + '_' + plotId, fontsize=self.titleFontSize, y=self.titleYPosition)
    def plot1DGraph(self, axs, axsRowIdx, axsColIdx, xAxisValues, yAxisValues, colors_idx):
        axs[axsRowIdx][axsColIdx].plot(xAxisValues, yAxisValues, self.colors[colors_idx])
        axs[axsRowIdx][axsColIdx].grid(b=True)
        return axs

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
            plotableWeights = self.layer.getPlotable2DWeights()

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

    def setGraphXAxisLable(self, axs, axsRowIdx, axsColIdx, label):
        axs[axsRowIdx][axsColIdx].set_xlabel(label) 

    def setGraphYAxisLable(self, axs, axsRowIdx, axsColIdx, label):
        axs[axsRowIdx][axsColIdx].set_ylabel(label)
        

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
        
