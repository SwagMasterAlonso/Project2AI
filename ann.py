#!/usr/bin/python

from numpy import *
import numpy as np
import sys


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
    #List of weights
    weights = []
    #List of inputs
    inputs = []
    sumInputs = 0 
    idOfConnectedNodes = []

    def __init__(self, id, inputs, weights, connections):
        self.inputs = inputs
        self.weights = weights
        self.idNum = id
        self.idOfConnectedNodes = connections
        
    def sumPoints(self):
        sum = np.dot([row[0] for row in self.weights], self.inputs)    
        self.sumInputs = sum
        
    def computeG(self):
        return 1/(1+math.exp(-(self.sumInputs)))
    def fireSignal(self):
        temp = self.computeG()
        if (temp > 0.5):
            return 1
        else: 
            return 0
    def appendWeights(self, weight):
        self.weights.append(weight)

class NNetwork:
    #list of nodes
    nodes = []
    
    