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
import tensorflow as tf;
from keras import backend as K
from SingleStepRandomGenerator import SingleStepTSGenerator
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


windowSize = 30;
batchSize=64
optimizer = Adam()
factory = SingleStepGenerator.GeneratorFactory(shp=(windowSize,1), dopt = optimizer)
genModel, g_input = factory.createLSTM(batch_size=batchSize, look_back = windowSize);
o_input = Input(shape=[1]);

lossModel = factory.createLossModel(overallInput=g_input, generatorOutput=genModel(g_input), observed=o_input);



tsList= loadCsv('../data/GBPUSD.csv');
numRealSamples = int(math.floor((len(tsList)-windowSize-1)/float(batchSize))*batchSize);

 
trainingSet = SingleStepTrainingSetGenerator(windowSize = windowSize, 
                                   numRealSamples = numRealSamples, tsList = tsList);
x,y = trainingSet.create();




history = Helpers.LossHistory(lossModel, filename='singlestepgenerator.model');



plot_model(genModel, to_file='gen_model.png')
plot_model(lossModel, to_file='model.png')

print(x.shape);
#quit();


for i in range(400):
    lossModel.fit([x,y],  
                  batch_size=batchSize, epochs=1, verbose=1
                ,callbacks=[history], shuffle=False)
    lossModel.reset_states()
    
    print(i);




tsg = SingleStepTSGenerator(genModel, windowSize, 1000, np.array([x[1] for x in tsList]), batchSize);

plt.figure("Random generator");
plt.plot(trainingSet.denormalise(tsg.generate()));

genModel.reset_states();
predictions = genModel.predict(x, batch_size=batchSize)


plt.figure("Actual");

act = np.array([x for _,x in tsList]) 
plt.plot(act    );
plt.figure("Predicted means");

v1= predictions[0]+np.sqrt(predictions[1]);
v2= predictions[0]-np.sqrt(predictions[1]);
plt.hold(True);
plt.plot(trainingSet.denormalise(predictions[0]), 'r');
plt.plot(trainingSet.denormalise(v1), 'b');
plt.plot(trainingSet.denormalise(v2), 'b');
plt.plot(trainingSet.denormalise(y),'g');

plt.figure("Predicted variance");
plt.plot(predictions[1]);
plt.show();

