'''
Created on 2 Nov 2017

@author: phil
'''
import Descriminator
from SingleStepGenerator import *;
import numpy as np;
from  BrownianGenerator import BrownianModel 
from WhiteNoiseGenerator import WhiteNoiseModel
from GarchGenerator import Garch11Model
from CsvDataImport import loadCsv;
from keras.optimizers import Adam
import keras;
import matplotlib.pyplot as plt
import math;

import tensorflow as tf;
from keras import backend as K
from SingleStepRandomGenerator import SingleStepTSGenerator


config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
    
tsList= loadCsv('../data/GBPUSD.csv');


descrWindowSize = 8000;
generatorWindowSize = 10
batchSize = 128;


singleStepGenModel =keras.models.load_model('./singlestepgenerator.model', custom_objects={"NormalPDFLogLikelyhoodLayer": NormalPDFLogLikelyhoodLayer});
m = singleStepGenModel.get_layer("Generator_model");
m.summary();
#m = singleStepGenModel


tsg = SingleStepTSGenerator(m, generatorWindowSize, descrWindowSize, np.array([x[1] for x in tsList]), batchSize);

plt.plot(tsg.generate());
plt.figure();
#plt.hold(True);

plt.plot(tsg.inputData);

plt.show();



quit();


trainedNet = keras.models.load_model('./model.keras');


numRealSamples = len(tsList)-descrWindowSize;
numFakeSamples = numRealSamples

print("Num real samples")
print(numRealSamples);
print("Num Fake samples")
print(numFakeSamples);

generator = TrainingSetGenerator(windowSize = descrWindowSize, 
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

with open("results.csv","w") as file: 

    for strideX in range(0, dataSetSize ):
        file.write(str(yPrime[strideX][0]))
        file.write(',');
        file.write(str(yPrime[strideX][1]))
        file.write(',');
        file.write(str(y[strideX][0]))
        file.write(',');
        file.write(str(y[strideX][1]))
        file.write(',');
        if generatorsUsed[strideX]!=None:
            file.write(type(generatorsUsed[strideX]).__name__)
        
        file.write('\n');
