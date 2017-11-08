'''
Created on 3 Nov 2017

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
import random;
import Helpers;
import Descriminator
from Generator import GeneratorFactory
from keras.models import Model
from keras import backend as K
import tensorflow as tf;
import TrainingSet;
#

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)


optimizer = Adam(lr=1e-4)
randomShape=25;
windowSize =1000;
useSavedDescriminator=False;

if useSavedDescriminator:
    descriminator = keras.models.load_model('./descriminator.keras');
else:
    descriminator = Descriminator.DescriminatorFactory((windowSize,1), 0.25, Adam(lr=1e-3)).create();

descriminator.summary();

generatorFactory = Generator.GeneratorFactory(dopt=optimizer, shp=[randomShape]);

Helpers.make_trainable(descriminator,False)
generator, g_input = generatorFactory.create();
fullGan = descriminator(generator(g_input));

GAN = Model(g_input, fullGan)
GAN.compile(loss='categorical_crossentropy', optimizer=optimizer)
GAN.summary()
generatorNN = GAN.get_layer("Generator_model");
tsList= loadCsv('../data/GBPUSD.csv');

samples =len(tsList)-windowSize-1

y = np.zeros((samples, 2));
for z in range(0, samples ):
    y[z][0] = 1;

# history = Helpers.LossHistory(generator, filename='generator.model')
# descHistory = Helpers.LossHistory(generator, filename='descriminator2.model')
noise_gen = np.random.uniform(0,1,size=[samples,randomShape])

ganTrainingSet = TrainingSet.GANTrainingSetGenerator(windowSize=windowSize, numRealSamples=samples, numFakeSamples=samples, tsList=tsList)

# Also make a descriminator only set
trainingSet = TrainingSet.TrainingSetGenerator(windowSize = windowSize, 
                                   numRealSamples = samples,
                                   numFakeSamples = samples);
descrX,descrY, _ = trainingSet.create(tsList);


for epoch in range (0,1000):
    
    print("Epoch " + str(epoch))
    print("Train generator");
    GAN.fit(noise_gen, y,  
           batch_size=128, epochs=2, verbose=1)
    yPrime = generatorNN.predict(noise_gen);
    xTrain, yTrain = ganTrainingSet.create(yPrime);
    
    print("Train descriminator");
    descriminator.fit(xTrain, yTrain, batch_size=128, epochs=1, verbose=1)
    
    print("Train descriminator with traditional training set");
    descriminator.fit(descrX, descrY, batch_size=128, epochs=1, verbose=1)
    
    GAN.save('gan_' + str(epoch) +".model");
    
    
