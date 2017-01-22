import numpy as numpy

def findSurvivors(data):
    numberPassengers = numpy.size(data.astype(numpy.float)) # find the number of passengers
    numberSurvived = numpy.sum(data.astype(numpy.float))    # find the number of survivors
    return numberSurvived / numberPassengers