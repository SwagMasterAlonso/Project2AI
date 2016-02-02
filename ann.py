#!/usr/bin/python
from numpy import *
import numpy as np
import sys

def main():
    # Reading in command line arguments.
    hiddenNodes = None
    holdout = None
    fileName = sys.argv[1]
    argc = len(sys.argv)
    if (argc == 4):
        if (sys.argv[2] == "h"):
            hiddenNodes = int(sys.argv[3])
            holdout = 20
        elif (sys.argv[2] == "p"):
            hiddenNodes = 5
            holdout = int(100*int(sys.argv[3]))
    elif (argc == 6):
        hiddenNodes = int(sys.argv[3])
        holdout = int(100*float(sys.argv[5]))
    else:
        hiddenNodes = 5
        holdout = 20
    count = 0
    matrixIndex = 0

    
    #Trys to open a file, outputs an error message otherwise.
    try:
        # Open a file
        with open(fileName, "r") as f:
            for line in f:
                count+=1
    except IOError:
        print "There was an error reading from", "hw5data.txt"
        sys.exit()
    count+=1
    
    #Creates two matrices with desired shape and initializes all cells to zero
    Xinput = np.zeros(shape=(count,2))
    answerMatrix =np.zeros(shape=(count,1)) 

    #Opens the file again to read in the data.
    try:
        # Open a file
        with open(fileName, "r") as f:
            for line in f:
                fields = line.strip().split()
                inputs = np.array([fields[0], fields[1]])
                Xinput[matrixIndex] = [fields[0],fields[1]]
                answerMatrix[matrixIndex] = [fields[2]]
                matrixIndex +=1
    except IOError:
        print "There was an error reading from", "hw5data.txt"
        sys.exit()

    # Close opened file
    f.close()    
    #Create a Neural Network using the desired parameters.
    NN = NNetwork(hiddenNodes,holdout,2,1,Xinput,answerMatrix)
    #forloop that continuously trains using the same training data a certain
    #number of iterations.
    for i in range (0,100):
        #Backpropagation learning algorithm
        NN.backPropogation(answerMatrix)

    #Setting the best weights based on the error rate at some iteration.
    NN.W1 = NN.BW1
    NN.W2 = NN.BW2
    #classifies the testing data and outputs error rate.
    ErrorRate = NN.classify(NN.createTestingSet())
    ErrorRate = ErrorRate*100
    sys.stdout.write("The Error Rate is "+str(ErrorRate)+"%")

#Neural Network class that takes in a data set, classifies this data by 
#forward propagation, and trains through the backprop algorithm.    
class NNetwork(object):
        #initialization method that sets the number of hidden nodes, output nodes
        #number of input nodes, the input data, and the answers to the input data.
        #Also saves the best weights after training and the squared error at some point in training.
        def __init__(self, h, p, inputNodes, outputNodes,inputData,answers):
            self.h = h
            self.p = p
            self.inputNodes = inputNodes
            self.outputNodes = outputNodes
            self.W1 = random.randint(3,size=(self.inputNodes, self.h))
            self.W2 = random.randint(3,size=(self.h, self.outputNodes))
            self.BW1 = None
            self.BW2 = None
            self.squaredErrorVal = 0.00000000000
            self.prevSquaredError = 0.0000000000
            self.classes = answers
            self.inputData = inputData

        #Forward propagation that outputs the classes of each example.
        def forwardFeed(self,data):    
            self.hiddenLayerMatrix = np.dot(data, self.W1)
            self.outputFromHiddenLayer = self.computeG(self.hiddenLayerMatrix)
            self.threshHold(self.outputFromHiddenLayer)
            self.outputMatrix = np.dot(self.outputFromHiddenLayer, self.W2)
            self.neuralNetOutput = self.computeG(self.outputMatrix)
            self.threshHold(self.neuralNetOutput)
            return self.neuralNetOutput
        
        #Returns the output of the sigmoid function.
        def computeG(self,data):

            return 1/(1+np.exp(-(data)))
        
        #Returns the output of the derivative of the sigmoid function.
        def computeGPrime(self,data):
            return np.exp(-data)/((1+np.exp(-data))**2)
        
        #Function that returns 1 if cell in the data matrix is
        #above 0.5, and 0 otherwise.
        def threshHold(self, data):
            data[data>.5] = 1
            data[data<=.5] = 0
            return data
        
        #Returns the testing set from the given input data
        #by separating the last p examples from the whole data set.
        def createTestingSet(self):
            testingSize = (self.p)*0.01*(len(self.inputData))
            testingData = np.array(self.inputData[len(self.inputData)-testingSize:len(self.inputData), 0:len(self.inputData)])
            return testingData
        
        #Separates the first n examples for the training set.
        #Returns this training set.
        def separateData(self, givenSet):
            trainingSize = len(givenSet) - (self.p)*0.01*(len(givenSet))
            trainingData = np.array(givenSet[0:trainingSize, 0:trainingSize])
            return trainingData
        
        #Given the testing data set, the Neural Network forward propagates these examples
        #and returns the error rate (correct/testingSetSize)
        def classify(self, givenDataset):
#             print"Begin testing of the data"
            #have forward feed return the output of the output layer
            #create error rate formula that takes this 1d array and subtracts from the 
            #answer list stored in NN.
            #output this error rate
            testingSize = (self.p)*0.01*(len(self.inputData))
            answers  = np.array(self.classes)
            #Gets the answers to the corresponding test examples.
            testAnswers = np.array(answers[(len(self.inputData) - len(givenDataset)):len(self.inputData), 0:len(givenDataset)])
            trainingAnswers = self.forwardFeed(givenDataset)
            numMisClassified = np.sum(np.subtract(testAnswers, trainingAnswers)**2)
            correct = len(trainingAnswers) - numMisClassified
            self.squaredErrorVal = (correct/testingSize)
            
            #Helps choose the best weights based the performance on the testing set.
            #Does not train the neural network on the testing data.
            if(self.squaredErrorVal>self.prevSquaredError):
                self.BW1 = self.W1
                self.BW2 = self.W2
                self.prevSquaredError = self.squaredErrorVal
            return correct/testingSize
        
        #Function that trains the Neural Network, by propagating the error back
        #up the layers and updating the weights.
        def backPropogation(self,classesSet):
            alpha = 0.03
            NNoutput = self.forwardFeed(self.separateData(self.inputData))
            expectedValues = self.separateData(classesSet)
            sub = expectedValues-NNoutput
            V = -1
            subMinus = V*np.array(sub)
            self.errorZ3 = np.multiply(subMinus,self.computeGPrime(self.outputMatrix))
            self.W2Error = np.dot(self.outputFromHiddenLayer.T,self.errorZ3)
            self.errorZ2 = np.dot(self.errorZ3,self.W2.T)*self.computeGPrime(self.hiddenLayerMatrix)
            self.W1Error = np.dot(self.separateData(self.inputData).T,self.errorZ2)

            #Updating the weights based on the error from the output.
            self.W1 = self.W1 - alpha*self.W1Error
            self.W2 = self.W2 - alpha*self.W2Error

            if(self.squaredErrorVal>self.prevSquaredError):
                self.BW1 = self.W1
                self.BW2 = self.W2
                self.prevSquaredError = self.squaredErrorVal    
            #Using the test set as a validation set.
            self.classify(self.createTestingSet()) 
        
        #Returns the squared error.
        def squaredError(self, givenTraining, givenAnswers):
#             print "Sum is : "
            return np.sum((givenAnswers - givenTraining)**2)*.5

#The main function
if __name__ == '__main__':
    main() 