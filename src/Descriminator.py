'''
Created on 30 Oct 2017

@author: phil
'''

from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten,MaxoutDense
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D, UpSampling1D,Convolution1D
from keras.layers.recurrent import LSTM
from keras.regularizers import *
from keras.layers.normalization import *
from keras.optimizers import *
from keras.datasets import mnist
from keras.models import Model



class DescriminatorFactory:
    def __init__(self, shp, dropout_rate, dopt):
        self.generatorRandomInputShape = shp;
        self.dropout_rate= dropout_rate;
        self.dopt = dopt; 
    
    
    def create(self):
        d_input = Input(shape=self.generatorRandomInputShape)
        H = Convolution1D(256,  5, subsample=2, border_mode = 'same', activation='relu')(d_input)
        H = LeakyReLU(0.2)(H)
        H = Dropout(self.dropout_rate)(H)
        H = Convolution1D(512,  5, subsample=2, border_mode = 'same', activation='relu')(H)
        H = LeakyReLU(0.2)(H)
        H = Dropout(self.dropout_rate)(H)
        H = Flatten()(H)
        H = Dense(256)(H)
        H = LeakyReLU(0.2)(H)
        H = Dropout(self.dropout_rate)(H)
        d_V = Dense(2,activation='softmax')(H)
        discriminator = Model(d_input,d_V)
        discriminator.compile(loss='categorical_crossentropy', optimizer=self.dopt)
        discriminator.summary()
        return discriminator
    

