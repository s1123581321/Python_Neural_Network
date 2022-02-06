import neuralNetworkFunctions as nNet
import numpy as np

#Network settings
weightRange = [-1, 1]
biasRange = [-1, 1]
#Input values the sigmoid of outputs
inputValues = []
testValues = []
expectedOutput = []
testExpectedOutput = []
for x in range(65):
    inputValues.append([(x*np.pi)/33])
    expectedOutput.append(np.array([nNet.activationFunction_F(np.sin(inputValues[x]))]))
    testValues.append([((2*x*np.pi)/33)])
    testExpectedOutput.append(np.array([nNet.activationFunction_F(np.sin(testValues[x]))]))


networkStructure = [len(inputValues[0]), 20, 20, len(expectedOutput[0])]

learningRate = 1
batchSize = len(inputValues)
epochs = 30000

nNet.runNetwork_F(networkStructure, weightRange, biasRange, epochs, batchSize, inputValues, expectedOutput, learningRate, testValues)
for x in testExpectedOutput:
    print(nNet.reverseSigmoid(x[0][0]))

