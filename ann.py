#!/usr/bin/python
from numpy import *
import numpy as np
import sys
from compiler.ast import nodes
    
    
    
    
    

    
    
    
    
    
    
    
    
def main():
    # Reading in command line arguments.
    fileName = sys.argv[1]
    hiddenNodes = int(sys.argv[2])
    print hiddenNodes
    
    holdout = int(sys.argv[3])
    print holdout

    inputs = []
    answers = []
    count = 0
    matrixIndex = 0

    
    
    try:
        # Open a file
        with open(fileName, "r") as f:
            for line in f:
               count+=1
    except IOError:
        print "There was an error reading from", "hw5data.txt"
        sys.exit()
    
    
    count+=1
    
    
    
    
    Xinput = np.zeros(shape=(count,2))
    answerMatrix =np.zeros(shape=(count,1)) 

    try:
        # Open a file
        with open(fileName, "r") as f:
            for line in f:
                fields = line.strip().split()
                inputs = np.array([fields[0], fields[1]])
                Xinput[matrixIndex] = [fields[0],fields[1]]
                answerMatrix[matrixIndex] = [fields[2]]
                matrixIndex +=1
               # inputs,tempArray
                #np.concatenate(Xinput,tempArray)
                #Xinput = np.vstack([Xinput, inputs])
                #answers.append(fields[2])
    except IOError:
        print "There was an error reading from", "hw5data.txt"
        sys.exit()
    print "Answer Matrix is:"
 #   print answerMatrix
    
    print "Reading from file finished."
  
    # Close opened file
    f.close()    
        
    NN = NNetwork(hiddenNodes,holdout,2,1,Xinput,answers)
   # NN.forwardFeed()
   # NN.separateData()
    NN.backPropogation(answerMatrix)
class NNetwork(object):
        
        def __init__(self, h, p, inputNodes, outputNodes,inputData,answers):
            self.h = h
            self.p = p
            self.inputNodes = inputNodes
            self.outputNodes = outputNodes
            self.W1 = np.random.randint(5, size=(self.inputNodes, self.h))
            self.W2 = np.random.randint(5, size=(self.h, self.outputNodes))
#             print "W1"
#             print self.W1
#             print "W2"
#             print self.W2
#             print "End of Weights"

            self.classes = answers
            self.inputData = inputData
        def forwardFeed(self,data):    
            self.hiddenLayerMatrix = np.dot(data, self.W1)
#             print "Start of hiddenLayerMat"
#             print self.hiddenLayerMatrix
#             print "End of hiddenLayerMat"
            self.outputFromHiddenLayer = self.computeG(self.hiddenLayerMatrix)
            #self.threshHold(self.outputFromHiddenLayer)
            self.outputMatrix = np.dot(self.outputFromHiddenLayer, self.W2)
            self.neuralNetOutput = self.computeG(self.outputMatrix)
            self.threshHold(self.neuralNetOutput)
#             print self.neuralNetOutput
            return self.neuralNetOutput
        def computeG(self,data):
            #print data
            return 1/(1+np.exp(-(data)))
        def computeGPrime(self,data):
            return np.exp(-data)/((1+np.exp(-data))**2)
        def threshHold(self, data):
            data[data>.5] = 1
            data[data<=.5] = 0
            return data
        def separateData(self, givenSet):
            trainingSize = len(givenSet) - (self.p)*0.01*(len(givenSet))
            trainingData = np.array(givenSet[0:trainingSize, 0:trainingSize])
            print "Shape of Train"
            print np.shape(trainingData)

            return trainingData
        def classify(self, givenDataset):
            print"Begin testing of the data"
            #have forward feed return the output of the output layer
            #create error rate formula that takes this 1d array and subtracts from the 
            #answer list stored in NN.
            #output this error rate
            testingSize = (self.p)*0.01*(len(self.inputData))
            answers  = np.array(self.answers)
            testAnswers = np.array(answers[0:len(givenDataset), 0:len(givenDataset)])
            trainingAnswers = self.forwardfeed()
            
            numMisClassified = np.sum(np.subtract(testAnswers, trainingAnswers))
            correct = len(trainingAnswers) - numMisClassified
            
            return correct/testingSize
            
        def backPropogation(self,classesSet):
            alpha = 0.2
            NNoutput = self.forwardFeed(self.separateData(self.inputData))
            expectedValues = self.separateData(classesSet)
            
            print "NNoutput"
            print NNoutput
            sub = expectedValues-NNoutput
            print "Starting Sub"
           # print sub
            V = -1
            subMinus = V*np.array(sub)
            self.errorZ3 = np.multiply(subMinus,self.computeGPrime(self.outputMatrix))
            self.W2Error = np.dot(self.outputFromHiddenLayer.T,self.errorZ3)
            print np.shape(self.errorZ3)
            print np.shape(self.W2.T)
            print np.shape(self.hiddenLayerMatrix)
            self.errorZ2 = np.dot(self.errorZ3,self.W2.T)*self.computeGPrime(self.hiddenLayerMatrix)
           
           
            print np.shape(self.inputData.T)
            print np.shape(self.errorZ2)

            
            self.W1Error = np.dot(self.separateData(self.inputData).T,self.errorZ2)
            print "W1Error"
            print self.W1Error
            print "W2Error"
            print self.W2Error
            self.W1 = self.W1 + alpha*self.W1Error
            self.W2 = self.W2 + alpha*self.W2Error
            print "W1Error"
            print self.W1
            print "W2Error"
            print self.W2
            print "Squared Error is: ",self.squaredError(NNoutput, expectedValues)
            return self.squaredError(NNoutput, expectedValues)
        
        def squaredError(self, givenTraining, givenAnswers):
#             print "Sum is : "
            return np.sum((givenAnswers - givenTraining)**2)*.5
            
if __name__ == '__main__':
    main() 