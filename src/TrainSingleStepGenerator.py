'''
Created on 10 Nov 2017

@author: phil
'''


import SingleStepGenerator

import numpy as np;
from CsvDataImport import loadCsv;
from keras.optimizers import Adam
import keras;
import matplotlib.pyplot as plt
from TrainingSet import SingleStepTrainingSetGenerator
import Helpers;
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model
import math
windowSize = 100;

optimizer = Adam(lr=1e-3)
factory = SingleStepGenerator.GeneratorFactory(shp=(windowSize,1), dopt = optimizer)
genModel, g_input = factory.create();
o_input = Input(shape=[1]);

lossModel = factory.createLossModel(overallInput=g_input, generatorOutput=genModel(g_input), observed=o_input);

lossModel.summary();
batchSize=128



tsList= loadCsv('../data/GBPUSD.csv');
numRealSamples = int(math.floor((len(tsList)-windowSize-1)/float(batchSize))*batchSize);

 
trainingSet = SingleStepTrainingSetGenerator(windowSize = windowSize, 
                                   numRealSamples = numRealSamples, tsList = tsList);
x,y = trainingSet.create();




history = Helpers.LossHistory(genModel, filename='singlestepgenerator.model');

plot_model(genModel, to_file='gen_model.png')
plot_model(lossModel, to_file='model.png')


lossModel.fit([x,y],  
              batch_size=batchSize, epochs=10, verbose=1)
#, callbacks=[history])


predictions = genModel.predict(x)



print(x[0]);
print(predictions[0][0])
print(predictions[1][0])

plt.figure("Actual");
plt.plot(np.array([x for _,x in tsList])     );
plt.figure("Predicted means");
plt.plot(predictions[0]);
plt.figure("Predicted variance");
plt.plot(predictions[1]);
plt.show();

