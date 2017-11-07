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

descriminator = keras.models.load_model('./descriminator.keras');

descriminator.summary();
optimizer = Adam(lr=1e-4)
randomShape=100;
ntrain = 10000
windowSize =1000;

generatorFactory = Generator.GeneratorFactory(dropout_rate=0.25,dopt=optimizer, shp=[randomShape]);

Helpers.make_trainable(descriminator,False)
generator, g_input = generatorFactory.create();
fullGan = descriminator(generator(g_input));

GAN = Model(g_input, fullGan)
GAN.compile(loss='categorical_crossentropy', optimizer=optimizer)
GAN.summary()
generatorNN = GAN.get_layer("Generator_model");
tsList= loadCsv('../data/GBPUSD.csv');

y = np.zeros((ntrain, 2));
for z in range(0, ntrain ):
    y[z][0] = 1;

# history = Helpers.LossHistory(generator, filename='generator.model')
# descHistory = Helpers.LossHistory(generator, filename='descriminator2.model')
noise_gen = np.random.uniform(0,1,size=[ntrain,randomShape])

samples =len(tsList)-windowSize-1
gen = TrainingSet.GANTrainingSetGenerator(windowSize=windowSize, numRealSamples=samples, numFakeSamples=samples, tsList=tsList)

for epoch in range (0,100):
    
    print("Epoch " + str(epoch))
    print("Train generator");
    GAN.fit(noise_gen, y,  
           batch_size=128, epochs=2, verbose=1)
    yPrime = generatorNN.predict(noise_gen);
    xTrain, yTrain = gen.create(yPrime);
    
    print("Train descriminator");
    descriminator.fit(xTrain, yTrain, batch_size=128, epochs=2, verbose=1)
    
    GAN.save('generator_gen' + str(epoch) +".model");
    
    
