#!/usr/bin/python

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
# Close opend file
f.close()

print fileName,hiddenNodes,holdout