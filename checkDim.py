import sys 
import math

stride = float(sys.argv[1])
poolingAdd1 = float(sys.argv[2])

def firstMethod(numInputs, numOutputs):
    print('First method')
    outputConv1 = math.ceil((numInputs - 400 + 1)/10.0)
    paddingOutput = outputConv1 + 40
    outputConv2 = math.ceil((paddingOutput - 40 + 1)/16.0)
    print(numOutputs, outputConv2)

def secondMethod(numInputs, numOutputs): 
    print('Second method')
    outputConv1 = math.ceil((numInputs - 400 + 1)) + poolingAdd1 * 2
    poolingOutput = math.ceil((outputConv1 - 560 + 1)/stride)
    paddingOutput = poolingOutput + 20 * 2 
    outputConv2 = math.ceil((paddingOutput - 40 + 1))
    print(numOutputs, outputConv2)

def runBoth(numInputs, numOutputs):
#    firstMethod(numInputs, numOutputs)
    secondMethod(numInputs, numOutputs)

runBoth(76240, 475)
runBoth(88720, 553)
runBoth(127760, 797)
runBoth(116400, 726)

    

