#!/usr/bin/env python3
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.signal import find_peaks
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
        self.epochRangeToPlot = [self.numEpochs + int(x) + 1 if int(x) < 0 else int(x) for x in self.plottingConfigs['plotRange']]
        self.reverse = self.plottingConfigs['reverse']
        self.layerType = self.plottingConfigs['layerType']
        self.weights = self.loadWeights()
        self.layer = self.getLayer()
        self.plotter = Plotter(self.pathToAnalysisDir, self.plottingConfigs, self.layer, self.epochRangeToPlot)

    def loadWeights(self):
        weights = []
        for i in self.epochRangeToPlot:
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
        dimInput = int(self.plottingConfigs['dimInput'])

        if(isLayerWeightComposedOf1Subarrays):
            return FeedForwardLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings, isPlottingDomainLog, doPaddedFourierTransform, sampleRate)
        elif(isLayerWeightComposedOf2Subarrays):
            if(self.layerType == 'conv'):
                return Conv1DLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings, isPlottingDomainLog, doPaddedFourierTransform, sampleRate, dimInput) 
            elif(self.layerType == 'feed'):
                return FeedForwardLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings, isPlottingDomainLog, dimInput)

        elif(isLayerWeightComposedOf3Subarrays):
            return Conv2DLayer(self.weights, self.nameOfLayer, self.nameOfLayerPath, wishedPlottings) 

    def run(self):
        self.plotter.plot()
        print('...plotting graphs for ' + str(self.nameOfLayer) + ' done!')


class Layer(object):
    """This is an abstract class"""

    def __init__(self, weights, name, namePath, isPlottingDomainLog, sampleRate, dimInput):
        self.isPlottingDomainLog = isPlottingDomainLog
        self.name = name
        self.namePath = namePath
        self.weights = weights
        self.dimInput = dimInput
        self.numEpochs = len(self.weights)
        self.shape = self.weights[0].shape
        self.filterSize = int(self.shape[1]/self.dimInput)
        self.allowedPlottings = []
        self.plottingsToDo = []
        self.layerType = None
        self.dimInputIdx = 0
        self.domain = None
        self.sampleRate = sampleRate
    
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
        self.timeAxisFreq = np.arange(int(self.filterSize/2)) if not doPaddedFourierTransform else np.arange(int(self.sampleRate/2))
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
        self.plotableWeights = [[] for x in range(self.dimInput)]
        numEpochs = len(weights)
        for epochIdx in range(numEpochs):
            weightsToAppend = self.extractWeightToAppend(weights[epochIdx], weights[epochIdx].shape[0])
            for i in range(self.dimInput):
                self.plotableWeights[i].append(weightsToAppend[:,:,i])
        return self.plotableWeights

    def extractWeightToAppend(self, weights, numFilters):
        weightsToAppend = np.zeros((numFilters, self.filterSize, self.dimInput))
        for i in range(numFilters): 
            for j in range(self.filterSize):
                for k in range(self.dimInput):
                    filterDimIdx = self.dimInput * j + k
                    weightsToAppend[i,j,k] = weights[i, filterDimIdx]
        return weightsToAppend

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

class Peaks(object):

    def __init__(self, fourierWeights):
        self.fourierWeights = fourierWeights
        self.filterFunctions = {
                'narrowBand': self.filterNarrowBandFilters,
                'band': self.filterBandFilters,
                'cleanPeaks': self.filterCleanPeaks,
                'dirtyPeaks': self.filterDirtyPeaks
        }
        self.barAccuracies = [1,10,2]

    def getPeaks(self, epoch, filterFunctionString, dimInputIdx):
        filterWeights = self.fourierWeights[dimInputIdx][epoch]
        assert filterFunctionString in self.filterFunctions, '{} does not exist'.format(filterFunctionString)
        filterFn = self.filterFunctions[filterFunctionString]
        allPeaks = []
        for filterWeight in filterWeights:
            maxVal = np.max(filterWeight)
            peaks = filterFn(filterWeight, maxVal)
            allPeaks += self.transformToTuples(peaks, filterWeight)
        return sorted(allPeaks)
        
    def filterNarrowBandFilters(self, filterWeight, maxVal):
        peaks,_ = find_peaks(filterWeight, height=max(0.75*maxVal, 10), distance=200)
        return peaks if len(peaks) == 1 else []

    def filterBandFilters(self, filterWeight, maxVal):
        peaks,_ = find_peaks(filterWeight, height=max(0.90*maxVal, 8), distance=400)
        return peaks if len(peaks) == 1 else []

    def filterCleanPeaks(self, filterWeight, maxVal):
        peaks,_ = find_peaks(filterWeight, height=max(0.6*maxVal, 8), distance=400)
        return peaks

    def filterDirtyPeaks(self, filterWeight, maxVal):
        peaks,_ = find_peaks(filterWeight, height=max(0.6*maxVal, 5), distance=400)
        return peaks

    def transformToTuples(self, peaks, filterWeight):
        return [ (peak, filterWeight[peak]) for peak in peaks ]

class FeedForwardLayer(Layer):
    
    def __init__(self, weights, name, namePath, wishedPlottings, isPlottingDomainLog, doPaddedFourierTransform, sampleRate, dimInput):
        super(FeedForwardLayer, self).__init__(weights, name, namePath, isPlottingDomainLog, dimInput)
        self.numFilters = self.shape[0]
        assert weights[0].shape[0] == self.numFilters, "dim not correct"
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

    def __init__(self, weights, name, namePath, wishedPlottings, isPlottingDomainLog, doPaddedFourierTransform, sampleRate, dimInput):
        super(Conv1DLayer, self).__init__(weights, name, namePath, isPlottingDomainLog, sampleRate, dimInput)
        self.numFilters = self.shape[0]
        assert weights[0].shape[0] == self.numFilters, "dim not correct"
        self.addToAllowedPlottings(['1DWeightsSimpleAll','2DFilterStats','2DWeightsHeat'])
        self.setLayerType(type(self).__name__)
        self.createPlottingsToDo(wishedPlottings)
        self.processedWeights = ProcessedWeights(self.weights, self.filterSize, isPlottingDomainLog, self.dimInput, doPaddedFourierTransform, sampleRate)
        self.peaks = Peaks(self.processedWeights.plotableWeightsFreq)
        self.channelUsedForPermutation = 0 

    def getPlotable1DWeights(self):
        return self.processedWeights.getPlotable1DWeights(self.domain, self.dimInputIdx)

    def getPlotable2DWeights(self):
        return self.processedWeights.getPlotable2DWeights(self.domain)

    def getPlotableSorted2DWeights(self):
        return self.processedWeights.getPlotableSorted2DWeights()

    def getPeaks(self, epoch, filterFn):
        return self.peaks.getPeaks(epoch, filterFn)

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

    def __init__(self, pathToAnalysisDir, plottingConfigs, layer, epochRangeToPlot):
        self.layer = layer
        self.pathToAnalysisDir = pathToAnalysisDir
        self.plottingConfigs = plottingConfigs
        self.colors = self.plottingConfigs['colors']
        self.epochRangeToPlot = epochRangeToPlot
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
            if('2DFilterStats' in self.layer.plottingsToDo):
                if(self.layer.domain == 'freq'):
                    self.plot2DFilterStats()
            if('2DWeightsHeat' in self.layer.plottingsToDo):
                if(self.layer.domain == 'time'):
                    self.plot2DHeatWeights('unsorted')
                elif(self.layer.domain == 'freq'):
                    self.plot2DHeatWeights('sorted')
                    self.plot2DHeatWeights('unsorted')
            if('3DWeightsHeat' in self.layer.plottingsToDo):
                 self.plot3DHeatWeights()

    def plot2DFilterStats(self):
        assert self.plottingConfigs['pad'], 'Fourier weights should be padded!'
        filterFunctionsToPlot = len(self.plottingConfigs['filterFunctions'])
        numBarPlots = len(self.layer.peaks.barAccuracies)

        for epochIdx, epoch in enumerate(self.epochRangeToPlot):
            fig, axs = plt.subplots(numBarPlots, filterFunctionsToPlot, figsize=self.figSize)
            for filterFunctionIdx, filterFunction in enumerate(self.plottingConfigs['filterFunctions']):
                peaksToPlot = self.layer.peaks.getPeaks(epochIdx, filterFunction, self.layer.dimInputIdx)
                for barIdx, barAccuracy in enumerate(self.layer.peaks.barAccuracies):
                    self.plot2DStat(axs, barIdx, filterFunctionIdx, peaksToPlot, barAccuracy, filterFunction)
            plotId = '_epoch' + str(epoch) + '_dimIdx' + str(self.layer.dimInputIdx) + '_filterLength=' + str(self.layer.filterSize) + '_stats'
            self.savePlot(plt, plotId)
            self.setPlotTitle(plt, plotId)

    def plot2DStat(self, axs, barIdx, filterFunctionIdx, peaksToPlot, barAccuracy, filterFunction):
        maxFreq = int((self.layer.sampleRate+1)/2)
        if(barAccuracy == 1):
            xAxisValues = [ x[0] for x in peaksToPlot ]
            yAxisValues = [ x[1] for x in peaksToPlot ]
            axs[barIdx][filterFunctionIdx].plot(xAxisValues, yAxisValues, 'ro')
            axs[barIdx][filterFunctionIdx].grid(b=True)
            axs[barIdx][filterFunctionIdx].set_xticks(range(0,maxFreq, int((maxFreq+1)/10)))
            axs[barIdx][filterFunctionIdx].set_xlabel('Frequency')
            axs[barIdx][filterFunctionIdx].set_ylabel('Magnitude')
            axs[barIdx][filterFunctionIdx].set_title(filterFunction)
        else: 
            intervalLen = int((maxFreq+1)/barAccuracy)
            values = [ val[0] for val in peaksToPlot ]
            axs[barIdx][filterFunctionIdx].hist(values, bins=barAccuracy)
            axs[barIdx][filterFunctionIdx].set_xticks(range(0,maxFreq+1, intervalLen))
            axs[barIdx][filterFunctionIdx].set_xlabel('Frequency')
            axs[barIdx][filterFunctionIdx].set_ylabel('Count')

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
                
            plotId = '_epoch' + str(self.epochRangeToPlot[epochRangeIdx]) + '_' + self.layer.domain + '_dimIdx' + str(self.layer.dimInputIdx) + '_filterLength=' + str(self.layer.filterSize) + '_all'
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
        
        fig, axs = plt.subplots(self.layer.dimInput, len(self.epochRangeToPlot), figsize=self.figSize, sharex=True, sharey=True)

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

        for epochRangeIdx, plotableWeightPerEpoch in enumerate(plotableWeights):
            for dimInputIdx, plotableWeightPerDim in enumerate(plotableWeightPerEpoch): 
                maxVal = np.max(plotableWeightPerDim) if np.max(plotableWeightPerDim) > maxVal else maxVal
                minVal = np.min(plotableWeightPerDim) if np.min(plotableWeightPerDim) < minVal else minVal
                im = axs[dimInputIdx][epochRangeIdx].imshow(plotableWeightPerDim, origin='lower', aspect='auto', cmap=self.plottingConfigs['cmap'])
                colorInterval = self.plottingConfigs['colorInterval']
                if(colorInterval):
                    im.set_clim(colorInterval[0], colorInterval[1])
                axs[dimInputIdx][epochRangeIdx].set_ylabel(self.layer.domain + '_for_channel_' + str(dimInputIdx))
                axs[dimInputIdx][epochRangeIdx].set_xlabel('filterIdx_' + mode + '_for epoch' + '_' + '%03d' % (self.epochRangeToPlot[epochRangeIdx],))

        fig.subplots_adjust(hspace=0)
        fig.subplots_adjust(right=0.8)
        plt.setp([a.get_xticklabels() for a in fig.axes[:-1]], visible=False)

        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(im, cax=cbar_ax)
        
        numDomain = '_log_applied' if(self.layer.isPlottingDomainLog and self.layer.domain == 'freq') else ''

        plt.suptitle(self.title + '_' + self.layer.domain + '_for_epoch_' + '_'.join(str(x) for x in self.epochRangeToPlot) + numDomain, fontsize=self.titleFontSize, y=self.titleYPosition)
        plt.savefig(self.pathToAnalysisDir + '/' + self.layer.namePath + '_heat_map_' + self.layer.domain + '_' + mode + numDomain + '_filterLength=' + str(self.layer.filterSize))

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
            fig, axs = plt.subplots(self.samplesPerRow, len(self.epochRangeToPlot), figsize=self.figSize) 
            
            for epochRangeIdx, epoch in enumerate(self.epochRangeToPlot):
                for i in range(self.samplesPerRow):
                    filterNum = self.samplesPerRow*plotIdx + i
                    axs[i][epochRangeIdx].plot(timeArray, plotableWeights[epoch-1][filterNum],self.colors[i])
                    axs[i][epochRangeIdx].grid()
                    axs[i][epochRangeIdx].set_ylabel('filter.' + "%02d" % (filterNum+1,) + '_epoch.'+ "%03d" % (epoch,) + '_weights')
                    axs[i][epochRangeIdx].set_xlabel(self.layer.domain) 

            nameToSave = self.layer.namePath + '_' + self.layer.domain + '_' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow)

            plt.suptitle(self.title + '_' + str(plotIdx*self.samplesPerRow+1) + '_to_' + str((plotIdx+1)*self.samplesPerRow), fontsize=self.titleFontSize, y=self.titleYPosition)
            plt.savefig(simple1DPlotDir + '/' + nameToSave)
        
