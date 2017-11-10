'''
Created on 2 Nov 2017

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
import tensorflow as tf;
from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)
K.set_session(session)

trainedNet = keras.models.load_model('./gan_306.model');

trainedNet.summary();
ganTrainingSet = trainedNet.get_layer("Generator_model");

randomShape=25;
ntrain = 1000
x = np.random.uniform(0,1,size=[ntrain,randomShape])


yPrime = ganTrainingSet.predict(x);
acc = trainedNet.predict(x);

for i in range(0,ntrain):
    
    series = yPrime[i];
    if(acc[i][0]>0.9):
        plt.figure(str(acc[i]));
        plt.plot(series);
        plt.show();

    

