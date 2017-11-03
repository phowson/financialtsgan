'''
Created on 2 Nov 2017

@author: phil
'''
import Descriminator
import Generator
import numpy as np;
from  BrownianGenerator import BrownianModel 
from WhiteNoiseGenerator import WhiteNoiseModel
from GarchGenerator import Garch11Model
from CsvDataImport import loadCsv;
from keras.optimizers import Adam
import keras;
import matplotlib.pyplot as plt

from TrainingSet import TrainingSetGenerator

tsList= loadCsv('../data/GBPUSD.csv');

trainedNet = keras.models.load_model('./model.keras');
windowSize = 1000;



numRealSamples = len(tsList)-windowSize;
numFakeSamples = numRealSamples

print("Num real samples")
print(numRealSamples);
print("Num Fake samples")
print(numFakeSamples);

generator = TrainingSetGenerator(windowSize = windowSize, 
                                   numRealSamples = numRealSamples,
                                   numFakeSamples = numFakeSamples);
dataSetSize = numRealSamples+numFakeSamples;                                    
x, y, generatorsUsed = generator.create(tsList);
                                   
yPrime = trainedNet.predict(x);


errorsPerGenerator = dict();
correct = 0;
for i in range(0, dataSetSize ):
    predictedReal = False;
    isReal = False;
    if yPrime[i][0]>yPrime[i][1]:
        predictedReal = True;
    
    if y[i][0]>y[i][1]:
        isReal = True;
    
    if isReal == predictedReal:
        correct = correct +1;
    else:
        print("-----")
        print("Did not predict correctly at index " + str(i))
        print("Generator used = ")
        print(generatorsUsed[i]);
                
        
        print("Predicted Y:")
        print(yPrime[i]);
        print("Actual Y:")
        print(y[i]);
        print("Is real? " + str(isReal))
        
        
        if generatorsUsed[i]!=None:
            name = type(generatorsUsed[i]).__name__;
            if not name in errorsPerGenerator.keys():
                errorsPerGenerator[name]=0;
            errorsPerGenerator[name]=errorsPerGenerator[name]+1;
        
        #plt.figure("Failed to predict");
        #plt.plot(x[i]);
        #plt.show();

a = float(correct) / float(dataSetSize)
print("Prediction accuracy = " + str(a));
print("Failed to catch " + str(dataSetSize-correct) +" timeseries")

print(errorsPerGenerator);

with open("results.txt","w") as file: 

    for strideX in range(0, dataSetSize ):
        file.write(str(yPrime[strideX][0]))
        file.write(',');
        file.write(str(yPrime[strideX][1]))
        file.write(',');
        file.write(str(y[strideX][0]))
        file.write(',');
        file.write(str(y[strideX][1]))
        
        file.write('\n');
