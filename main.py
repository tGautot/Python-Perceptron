from random import random

def normalizeValue(val):
    if val < 0:
        return -1
    else:
        return 1

class Perceptron:

    def __init__(self, inputSize):
        self.inputLayerSize = inputSize
        self.weights = []
        self.bias = random()*10 - 5

        for i in range(inputSize):
            self.weights.append(random()*10 - 5)

    def trainOnInput(self, inputVals, expectedOutputVal, learningRate):
        nnVal = self.processInput(inputVals)
        error = expectedOutputVal - nnVal
        self.adjustForError(inputVals, error, learningRate)

    def processInput(self, nnInput):
        assert len(nnInput) == len(self.weights)
        unprocessedOutputVal = self.bias

        for i in range(len(nnInput)):
            unprocessedOutputVal += nnInput[i]*self.weights[i]
        
        return normalizeValue(unprocessedOutputVal)

    def adjustForError(self, inputVals, error, learningRate):
        for i in range(len(self.weights)):
            self.weights[i] += error*inputVals[i]*learningRate
        self.bias += error*learningRate


# Try to have the perceptron guess if a point is above (1)
# or below (-1) the line given by the equation

def lineY(x):
    return 1.6*x + 3.4

def generateTrainingSet(trainingSize):
    trainingSet = []
    for i in range(trainingSize):
        x = random()*20 - 10
        y = random()*20 - 10
        if y > lineY(x):
            output = 1
        else:
            output = -1
    
        trainingSet.append([x, y, output])
    return trainingSet


perceptron = Perceptron(2)
trainingRate = 0.1
trainingSetSize = 1000000
trainingSet = generateTrainingSet(trainingSetSize)

testSetSize = 1000
testSet = generateTrainingSet(testSetSize)

score = 0
for test in testSet:
    if perceptron.processInput([test[0], test[1]]) == test[2]:
        score += 1

print("Score before training {}/{}".format(score, testSetSize))

for j in range(trainingSetSize):
    perceptron.trainOnInput([trainingSet[j][0], trainingSet[j][1]], 
                             trainingSet[j][2], trainingRate)

score = 0
for test in testSet:
    if perceptron.processInput([test[0], test[1]]) == test[2]:
        score += 1

print("Score after training {}/{}".format(score, testSetSize))


