# Exploring the Titanic through Machine Learning in Python
# --------------------------------------------------------
# The purpose of this project is to use machine learning
# techniques in python to predict the survival rate of the
# passengers of the titanic.

##########################
###### DEPENDENCIES ######
##########################
import os as os        # Lib for pathing to data files
import csv as csv      # Lib for reading csv files
import numpy as numpy  # Lib to provide ML functions
import DataUtils as dataUtils  # Collection of functions defined to group common functions

#########################
##### READ IN FILES #####
#########################
scriptDir = os.path.dirname(__file__) #dynamically get current directory
trainingRelPath = "Data/train.csv"    # Relative path to training data
testRelPath = "Data/test.csv"         # Relative path to test data

# Construct full paths
trainingDataFilePath = os.path.join(scriptDir, trainingRelPath)
testDataFilePath = os.path.join(scriptDir, testRelPath)

# Read files in using with pattern to open /close files safely
with open(trainingDataFilePath, 'rt') as f:  # open training data
    trainingFileObject = csv.reader(f)       # read the file in
    header = trainingFileObject.__next__()   # skip the header
    trainingData = []                        # variable to store the training data
    for row in trainingFileObject:           # read each row into data array
        trainingData.append(row)             # append each row
    trainingData = numpy.array(trainingData) # Convert to an array

with open(testDataFilePath, 'rt') as f:      # open test data
    testFileObject = csv.reader(f)
    header = testFileObject.__next__()
    testData = []
    for row in testFileObject:
        testData.append(row)
    testData = numpy.array(testData)


#################################
##### PROCESS TRAINING DATA #####
#################################
proportionSurvivors = dataUtils.findSurvivors(trainingData[0::,1])  # find the percentage that lived
womenStats = trainingData[0::, 4] == "female"   # find all the women in the data
menStats = trainingData[0::, 4] != "female"     # find all the men in the data

# Calculate survival rates
womenSurvivors = dataUtils.findSurvivors(trainingData[womenStats, 1])
menSurvivors = dataUtils.findSurvivors(trainingData[menStats, 1])

# Print Data
print('Proportion of women who survived is %s' % womenSurvivors)
print('Proportion of men who survived is %s' % menSurvivors)

#############################
##### PROCESS TEST DATA #####
#############################
outputRelPath = 'Data/genderBasedModel.csv'             # Relative output data dir
outputFilePath = os.path.join(scriptDir, outputRelPath) # Absolute output data dir

# Open the output file
with open(outputFilePath, 'wt') as predictionFile:

    predictionFileObject = csv.writer(predictionFile)

    # Write results to file based on test data
    predictionFileObject.writerow(["PassengerId", "Survived"]) # write the headers for the output file

    # Iterate over the test data and write the results to file
    for row in testData:
        if row[3] == "female":
            predictionFileObject.writerow([row[0], "1"]) # predict that women survive
            #print([row,"1"])
        else:
            predictionFileObject.writerow([row[0], "0"]) # predict that men die
            #print([row,"0"])


            # Changes 4 real