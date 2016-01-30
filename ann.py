#!/usr/bin/python
from numpy import *
import numpy as np
import sys
from compiler.ast import nodes
from objc._objc import NULL
    
    
    
    
    
def main():
    readFromCommand()
    
def readFromCommand():
    # Reading in command line arguments.
    fileName = sys.argv[1]
    hiddenNodes = int(sys.argv[2])
    holdout = int(sys.argv[3])
    inputs = []
    answers = []
    try:
        # Open a file
        with open(fileName, "r") as f:
            for line in f:
                fields = line.strip().split()
                tempArray = np.array([fields[0], fields[1]])
                inputs.append(tempArray)
                answers.append(fields[2])
    except IOError:
        print "There was an error reading from", "hw5data.txt"
        sys.exit()
      
    print "Reading from file finished."
    print inputs
    print answers
    # Close opened file
    f.close()
    
    # print fileName,hiddenNodes,holdout
    
class NNetwork(object):
        
        def __init__(self, h, p, inputNodes, outputNodes):
            self.h = h
            self.p = p
            self.inputNodes = inputNodes
            self.outputNodes = outputNodes
            self.W1 = np.random.rand(5, size=(self.inputNodes, self.hiddenNodes))
            self.W2 = np.random.rand(5, size=(self.hiddenNodes, self.outputNodes))
            self.classes = None;
        def forwardFeed(self):    
            self.hiddenLayerMatrix = np.dot(self.inputNodes, self.W1)
            self.outputFromHiddenLayer = self.comeputeG(self.hiddenLayerMatrix)
            self.outputMatrix = np.dot(self.outputFromHiddenLayer, self.W2)
            self.neuralNetOutput = self.computeG(self.outputMatrix)
        
if __name__ == '__main__':
    main() 