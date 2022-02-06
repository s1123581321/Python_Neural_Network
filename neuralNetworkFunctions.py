import numpy as np
import copy

activations = []
zValues = []
weights = []
biases = []
costReturn = []
errors = []
biasChanges = []
weightChanges = []
averageChangeWeights = []
averageChangeBiases = []


def networkArrays_F(networkStructureF, inputValuesF):
    global activations
    inputValues = []
    for x in inputValuesF:
        inputValues.append([x])
    activations = [np.array(inputValues)]
    global errors
    errors = []
    for x in networkStructureF[1:]:
        activations.append(np.zeros((x, 1)))
        errors.append(np.zeros((x, 1)))
    global zValues
    zValues = copy.deepcopy(activations)


def parameterArrays_F(networkStructureF, weightRangeF, biasRangeF):
    global weights
    weights = []
    for x in zip(networkStructureF[1:], networkStructureF[:-1]):
        weights.append(np.random.uniform(weightRangeF[0], weightRangeF[1], x))
    global biases
    biases = []
    for x in networkStructureF[1:]:
        biases.append(np.random.uniform(biasRangeF[0], biasRangeF[1], (x, 1)))


def feedforward_F(layers):
    for x in range(1, layers):
        zValues[x] = np.matmul(weights[x-1], activations[x-1]) + biases[x-1]
        activations[x] = activationFunction_F(zValues[x])


def activationFunction_F(x):
    return 1.0/(1.0+np.exp(-x))


def activationDerivative_F(x):
    return np.true_divide(np.exp(-x), (1+np.exp(-x))**2)


def costFunction_F(expectedOutputF):
    global costReturn
    costReturn = 0.5*np.sum((expectedOutputF-activations[-1])**2)


def costDerivative_F(expectedOutputF):
    return expectedOutputF-activations[-1]


def errorPropagation_F(expectedOutputF):
    errors[-1] = costDerivative_F(expectedOutputF)*activationDerivative_F(zValues[-1])
    for x in reversed(range(1, len(errors))):
        errors[x-1] = np.matmul(np.transpose(weights[x]), errors[x])*activationDerivative_F(zValues[x])


def parameterChanges():
    global biasChanges
    biasChanges = copy.deepcopy(errors)
    global weightChanges
    weightChanges = []
    for x in range(len(errors)):
        weightChanges.append(np.outer(errors[x], activations[x]))
    for x in range(len(biasChanges)):
        averageChangeBiases[x] = np.add(averageChangeBiases[x], biasChanges[x])
        averageChangeWeights[x] = np.add(averageChangeWeights[x], weightChanges[x])


def averageChangesInitialisation_F(networkStructureF):
    global averageChangeWeights
    averageChangeWeights = []
    for x in zip(networkStructureF[1:], networkStructureF[:-1]):
        averageChangeWeights.append(np.zeros(x))
    global averageChangeBiases
    averageChangeBiases = []
    for x in networkStructureF[1:]:
        averageChangeBiases.append(np.zeros((x, 1)))


def changeParameters(learningRateF, batchSizeF):
    for x in range(len(weights)):
        weights[x] = np.add(weights[x], learningRateF*np.true_divide(averageChangeWeights[x], batchSizeF))
        biases[x] = np.add(biases[x], learningRateF*np.true_divide(averageChangeBiases[x], batchSizeF))


def runNetwork_F(networkStructureF, weightRangeF, biasRangeF, epochsF, batchSizeF, inputValuesF, expectedOutputF, learningRateF, testValuesF):
    parameterArrays_F(networkStructureF, weightRangeF, biasRangeF)
    testFunction(networkStructureF, batchSizeF, testValuesF)
    for y in range(epochsF):
        averageChangesInitialisation_F(networkStructureF)
        for x in range(batchSizeF):
            networkInput = inputValuesF[x]
            networkOutput = expectedOutputF[x]
            #networkArrays_F(networkStructureF, inputValuesF)
            networkArrays_F(networkStructureF, networkInput)
            feedforward_F(len(networkStructureF))
            #errorPropagation_F(expectedOutputF)
            errorPropagation_F(networkOutput)
            parameterChanges()
        changeParameters(learningRateF, batchSizeF)
        if y == 999:
            testFunction(networkStructureF, batchSizeF, testValuesF)
        if y == 9999:
            testFunction(networkStructureF, batchSizeF, testValuesF)
    testFunction(networkStructureF, batchSizeF, testValuesF)


def testFunction(networkStructureF, batchSizeF, inputValuesF):
    for x in range(batchSizeF):
        networkInput = inputValuesF[x]
        networkArrays_F(networkStructureF, networkInput)
        feedforward_F(len(networkStructureF))
        print(reverseSigmoid(activations[-1][0][0]))
    print()


def reverseSigmoid(x):
    return -np.log((1/x)-1)
