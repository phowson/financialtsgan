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
import SingleStepGenerator
import tensorflow as tf;
from keras import backend as K
from SingleStepRandomGenerator import SingleStepTSGenerator
from TrainingSet import SingleStepTrainingSetGenerator
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)
    
tsList= loadCsv('../data/GBPUSD.csv');


descrWindowSize = 100;
generatorWindowSize = 20
batchSize = 128;
optimizer = Adam()

trainingSet = SingleStepTrainingSetGenerator(windowSize = generatorWindowSize, 
                                   numRealSamples = descrWindowSize, tsList = tsList);

singleStepGenModelFactory = SingleStepGenerator.GeneratorFactory(shp=(generatorWindowSize,1), dopt = optimizer)
genModel, g_input = singleStepGenModelFactory.createLSTM(batch_size=batchSize, look_back = generatorWindowSize);
o_input = Input(shape=[1]);
lossModel = singleStepGenModelFactory.createLossModel(overallInput=g_input, generatorOutput=genModel(g_input), observed=o_input);



#m = singleStepGenModel

lossModel.load_weights('singlestepgenerator.model')
tsg = SingleStepTSGenerator(genModel, generatorWindowSize, descrWindowSize, np.array([x[1] for x in tsList]), batchSize);


plt.figure('Random data');
plt.plot(trainingSet.denormalise(tsg.generate()));
plt.figure('Real data');
#plt.hold(True);

plt.plot(trainingSet.denormalise(tsg.inputData));

plt.show();

