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
import Helpers;



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


history = Helpers.LossHistory(descrModel, filename='descriminator.model');
descrModel.fit(x, y,  
              batch_size=512, epochs=500, verbose=1, callbacks=[history])


