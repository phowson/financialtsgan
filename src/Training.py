'''
Created on 30 Oct 2017

@author: phil
'''

import Descriminator
import Generator
import numpy as np;
from CsvDataImport import loadCsv;
from keras.optimizers import Adam
import keras;
import matplotlib.pyplot as plt
from TrainingSet import TrainingSetGenerator

class LossHistory(keras.callbacks.Callback):
    
    def __init__(self, model):
        self.model = model;
        self.minLoss = 9999e999;
    

    def on_epoch_end(self, batch, logs={}):
        l = logs.get('loss');        
        if (l<self.minLoss):
            print("Saving new best model");
            self.model.save("model.keras")
            self.minLoss = l;



windowSize = 1000;

optimizer = Adam(lr=1e-3)
factory = Descriminator.DescriminatorFactory((windowSize,1), 0.25, optimizer)
descrModel = factory.create();

tsList= loadCsv('../data/GBPUSD.csv');

numRealSamples = len(tsList)-windowSize-windowSize*2;
numFakeSamples = numRealSamples
dataSetSize = numRealSamples+numFakeSamples;
 
trainingSet = TrainingSetGenerator(windowSize = windowSize, 
                                   numRealSamples = numRealSamples,
                                   numFakeSamples = numFakeSamples);
x,y, generatorsUsed = trainingSet.create(tsList);


history = LossHistory(descrModel);
descrModel.fit(x, y,  
              batch_size=256, epochs=500, verbose=1, callbacks=[history])


yPrime = descrModel.predict(x);


with open("results.txt","w") as file: 

    for strideX in range(0, dataSetSize ):
        file.write(str(yPrime[strideX]))
        file.write('\n');


