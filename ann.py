#!/usr/bin/python

from numpy import *
import sys


#Reading in command line arguments.
fileName = sys.argv[1]
hiddenNodes = int(sys.argv[2])
holdout = int(sys.argv[3])


try:
    # Open a file
    with open("hw5data.txt", "r") as f:
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

class InputNode:

    def __init__(self, point):
        self.x = point.x
        self.y = point.y

class OutputNode:
    #List of inputs
    inputs = []
    sumInputs = 0 

    def __init__(self, inputs):
        self.inputs = inputs
        
    def sumPoints(self):
        for point in self.inputs:
            sum += point.x + point.y
            
        self.sumInputs = sum
    def computeG(self):
        return 1/(1+math.exp(-(self.sumInputs)))
    def computeClass(self):
        temp = self.computeG()
        if (temp > 0.5):
            return 1
        else: 
            return 0

class HiddenNode:
    #List of inputs
    inputs = []
    sumInputs = 0 

    def __init__(self, inputs):
        self.inputs = inputs
        
    def sumPoints(self):
        for point in self.inputs:
            sum += point.x + point.y
            
        self.sumInputs = sum
    def computeG(self):
        return 1/(1+math.exp(-(self.sumInputs)))

class Point:

    def __init__(self, x, y):
        self.x = x
        self.y = y

