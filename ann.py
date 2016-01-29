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
    
    
    
    
    
    class NNetwork(object):
      
        
        
        
        
        def__init__(self,h,p,inputNodes,outputNodes):
            
            self.h = h
            self.p = p
            self.inputNodes = inputNodes
            self.outputNodes = outputNodes
            W1 = np.random.rand(5, size =(self.inputNodes,self.hiddenNodes))
            W2 = np.random.rand(5, size =(self.hiddenNodes,self.outputNodes))
            
            
            
        
        
        
        def forwardFeed(self):

        
        
        
        
        
        
#         def __init__(self, weights, inputNodes, hiddenNodes, outputNode):
#             self.weights = weights
#             self.inputNodes = inputNodes
#             self.hiddenNodes = hiddenNodes
#             self.outputNode = outputNode
#             
#         def forwardFeed(self):
#             print()
#             #forward feed needs to call sum points
#             
#             
#               for j in range(0,len(hiddenNodes))
#                  for i in range(0,len(inputNodes)):
#                     tempSum = sumPoints(inputNodes[i])
#                     hiddenNodes[j].appendInputs(self.fireSignal(tempSum))    
#                 
#            
#             for j in range(0,len(inputNodes)):
#                 tempSumHidden = sumPoints(hiddenNodes[j])
#                 outputNode.appendOutputs(self.fireSignal(tempSumHidden)  
#         
#         
#         
#         
#         def computeG(self,numSum):
#             return 1/(1+math.exp(-(numSum)))
#         
#         def fireSignal(self,numSum):
#             
#             temp = self.computeG(numSum)
#             if (temp > 0.5):
#                 return 1
#             else: 
#                 return 0
#         
#         def sumPoints(self, nodeI):
#             sumInputs = 0
#             for i in nodeI.idOfConnectedNodes: 
#                 sumInputs += np.dot([col[i] for col in self.weights], [col[i] for col in nodeI.inputs])   
#                 return sumInputs
        