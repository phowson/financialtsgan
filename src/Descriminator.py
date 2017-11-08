'''
Created on 30 Oct 2017

@author: phil
'''

import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D, UpSampling1D,Convolution1D
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
from keras.models import Model



class DescriminatorFactory:
    def __init__(self, shp, dropout_rate, dopt):
        self.inputShape = shp;
        self.dropout_rate= dropout_rate;
        self.dopt = dopt; 
    
    
    def create(self):
        d_input = Input(shape=self.inputShape)
        H = d_input
        H = Conv1D(128,  kernel_size=5, strides=2, dilation_rate=1,  padding = 'same', activation='relu')(H)       
        H = LeakyReLU(0.1)(H)        
        H = Dropout(self.dropout_rate)(H)
        H = Conv1D(32,  kernel_size=3, strides=2, dilation_rate=1, padding = 'same', activation='relu')(H)
        H = LeakyReLU(0.1)(H)
        H = Dropout(self.dropout_rate)(H)        
        H = Flatten()(H)


        # Was single dense(256). 
        # Put a bottleneck in here to try and reduce overfitting.          
        H = Dense(16)(H)
        H = LeakyReLU(0.1)(H)
        H = Dense(8)(H)
        H = LeakyReLU(0.1)(H)
        
        H = Dropout(self.dropout_rate)(H)
        d_V = Dense(2,activation='softmax')(H)
        discriminator = Model(d_input,d_V)
        discriminator.compile(loss='categorical_crossentropy', optimizer=self.dopt)
        print(discriminator.summary())
        return discriminator
    

