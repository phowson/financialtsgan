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
from TrainingSet import TrainingSetGenerator,rolling_window, RangeNormaliser


class SingleStepTSGenerator:
    def __init__(self, model, windowSize, outWindowSize, inputData):
        self.model = model;
        normaliser = RangeNormaliser(inputData);
        
        self.windowSize = windowSize;
        self.outWindowSize = outWindowSize;
        self.genBuffer = np.zeros((windowSize));
        self.inputData = normaliser.normalise(inputData);
        
        
    def generate(self):
        rw = rolling_window(self.inputData, self.windowSize);
        startx = int(np.random.uniform() * (self.inputData.shape[0] - self.windowSize));
        self.genBuffer[0:] = rw[startx][:];
        gb = np.reshape(self.genBuffer, (1,self.windowSize,1));
        outBuffer = np.zeros((self.outWindowSize));
        self.model.summary();
        m = self.model.get_layer("Generator_model");
        
        for i in range(0, self.outWindowSize):
            p = m.predict(gb);
            rs = p[0][0] + np.random.normal()*math.sqrt(p[1][0]) ;
            outBuffer[i] = rs;
            
            for z in range(0, self.genBuffer.shape[0]-1):
                gb[0][z][0] = gb[0][z+1][0];
            gb[0][self.genBuffer.shape[0]-1][0] = rs;
            
            #plt.plot(self.genBuffer);
            #plt.show();
        return outBuffer;
    
tsList= loadCsv('../data/GBPUSD.csv');


descrWindowSize = 8000;
generatorWindowSize = 40


singleStepGenModel =keras.models.load_model('./singlestepgenerator.model', custom_objects={"NormalPDFLogLikelyhoodLayer": NormalPDFLogLikelyhoodLayer});
tsg = SingleStepTSGenerator(singleStepGenModel, generatorWindowSize, descrWindowSize, np.array([x[1] for x in tsList]));

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
