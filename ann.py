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
    
    try:
        # Open a file
        with open(fileName, "r") as f:
            for line in f:
                fields = line.strip().split()
                inputs = np.array([fields[0], fields[1]])
                Xinput[matrixIndex] = [fields[0],fields[1]]
                matrixIndex +=1
               # inputs,tempArray
                #np.concatenate(Xinput,tempArray)
                #Xinput = np.vstack([Xinput, inputs])
                answers.append(fields[2])
    except IOError:
        print "There was an error reading from", "hw5data.txt"
        sys.exit()
      
      
    
    print "Reading from file finished."
  
    # Close opened file
    f.close()    
        
    NN = NNetwork(hiddenNodes,holdout,2,1,Xinput,answers)
    NN.forwardFeed()
class NNetwork(object):
        
        def __init__(self, h, p, inputNodes, outputNodes,inputData,answers):
            self.h = h
            self.p = p
            self.inputNodes = inputNodes
            self.outputNodes = outputNodes
            self.W1 = np.random.randint(5, size=(self.inputNodes, self.h))
            self.W2 = np.random.randint(5, size=(self.h, self.outputNodes))
            print "W1"
            print self.W1
            print "W2"
            print self.W2
            print "End of Weights"

            self.classes = answers
            self.inputData = inputData
        def forwardFeed(self):    
            self.hiddenLayerMatrix = np.dot(self.inputData, self.W1)
            print "Start of hiddenLayerMat"
            print self.hiddenLayerMatrix
            print "End of hiddenLayerMat"
            self.outputFromHiddenLayer = self.computeG(self.hiddenLayerMatrix)
            self.outputMatrix = np.dot(self.outputFromHiddenLayer, self.W2)
            self.neuralNetOutput = self.computeG(self.outputMatrix)
            print self.neuralNetOutput
        def computeG(self,data):
            #print data
            return 1/(1+np.exp(-(data)))
        
if __name__ == '__main__':
    main() 