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
from keras.layers.normalization import BatchNormalization
from keras.optimizers import *
from keras.datasets import mnist
from keras.models import Model



class GeneratorFactory:
    def __init__(self, dropout_rate, dopt, shp=[100]):
        self.randomInputShape = shp;
        self.dropout_rate= dropout_rate;
        self.dopt = dopt; 
    
    
    def create(self):
        g_input = Input(shape=self.randomInputShape)
        nch = 200
        H = Dense(nch*14*14, init='glorot_normal')(g_input)
        H = BatchNormalization(mode=2)(H)
        H = Activation('relu')(H)
        H = Reshape( [nch, 14, 14] )(H)
        H = UpSampling1D(size=(2, 2))(H)
        H = Convolution1D(nch/2, 3, 3, border_mode='same', init='glorot_uniform')(H)
        H = BatchNormalization(mode=2)(H)
        H = Activation('relu')(H)
        H = Convolution1D(nch/4, 3, 3, border_mode='same', init='glorot_uniform')(H)
        H = BatchNormalization(mode=2)(H)
        H = Activation('relu')(H)
        H = Convolution1D(1, 1, 1, border_mode='same', init='glorot_uniform')(H)
        g_V = Activation('sigmoid')(H)
        generator = Model(g_input,g_V)
        generator.compile(loss='binary_crossentropy', optimizer=self.opt)
        generator.summary()
        return generator
    

