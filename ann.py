#!/usr/bin/python

from numpy import *
import numpy as np
import sys
from compiler.ast import nodes
from objc._objc import NULL


def readFromCommand():
    #Reading in command line arguments.
    fileName = sys.argv[1]
    hiddenNodes = int(sys.argv[2])
    holdout = int(sys.argv[3])
    
    
    try:
        # Open a file
        with open(fileName, "r") as f:
            for line in f:
                fields = line.strip().split()
                print fields[0], fields[1], fields[2] 
    except IOError:
        print "There was an error reading from", "hw5data.txt"
        sys.exit()
      
    print "Reading from file finished."
    # Close opened file
    f.close()
    
    # print fileName,hiddenNodes,holdout


class Node:
    #id number for node
    idNum = 0
    #List of inputs
    inputs = []
    sumInputs = 0 
    idOfConnectedNodes = []

    def __init__(self, id, inputs, connections):
        self.inputs = inputs
        self.idNum = id
        self.idOfConnectedNodes = connections
        
    def appendWeights(self, weight):
        self.weights.append(weight)



class NNetwork:
    #list of nodes
    inputNodes = []
    hiddenNodes = []
    outputNode = None
    #List of weights
    weights = []
    
    def __init__(self, weights, inputNodes, hiddenNodes, outputNode):
        self.weights = weights
        self.inputNodes = inputNodes
        self.hiddenNodes = hiddenNodes
        self.outputNode = outputNode
        
    def forwardFeed(self):
        print()
    def computeG(self):
        return 1/(1+math.exp(-(self.sumInputs)))
    
    def fireSignal(self):
        temp = self.computeG()
        if (temp > 0.5):
            return 1
        else: 
            return 0
    
    def sumPoints(self, nodeI):
        sumInputs = 0
        for i in nodeI.idOfConnectedNodes: 
            sumInputs += np.dot([col[i] for col in self.weights], [col[i] for col in nodeI.inputs])   
            return sumInputs
    