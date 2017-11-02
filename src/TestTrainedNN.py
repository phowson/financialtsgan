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

trainedNet = keras.models.load_model('./trained.keras');
windowSize = 1000;

numRealSamples = len(tsList)-windowSize;
numFakeSamples = numRealSamples
generator = TrainingSetGenerator(windowSize = windowSize, 
                                   numRealSamples = numRealSamples,
                                   numFakeSamples = numFakeSamples);
dataSetSize = numRealSamples+numFakeSamples;                                    
x, y = generator.create(tsList);
                                   
yPrime = trainedNet.predict(x);

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
        print(yPrime[i]);
        print(y[i]);
        print("Is real? " + str(isReal))
        plt.figure("Failed to predict");
        plt.plot(x[i]);
        plt.show();

a = float(correct) / float(dataSetSize)
print("Prediction accuracy = " + str(a));
print("Failed to catch " + str(dataSetSize-correct) +" timeseries")

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
