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

descriminator = keras.models.load_model('./model.keras');

descriminator.summary();
optimizer = Adam(lr=1e-3)
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


noise_gen = np.random.uniform(0,1,size=[ntrain,randomShape])

y = np.zeros((ntrain, 2));
for z in range(0, ntrain ):
    y[z][0] = 1;

history = Helpers.LossHistory(generator, filename='generator.model')

GAN.fit(noise_gen, y,  
       batch_size=128, epochs=500, verbose=1, callbacks=[history])
