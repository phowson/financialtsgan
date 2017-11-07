'''
Created on 30 Oct 2017

@author: phil
'''

from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge
from keras.layers.core import Reshape,Dense,Dropout,Activation,Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.activations import *
from keras.layers.wrappers import TimeDistributed
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Conv1D, MaxPooling1D, ZeroPadding1D, UpSampling1D,Convolution1D
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
    
    
    def create(self, nch = 125):
        g_input = Input(shape=self.randomInputShape)
        
        #
        H = Dense(nch*2, kernel_initializer="glorot_normal")(g_input)
        H = BatchNormalization()(H)
        H = Activation('relu')(H)
        H = Reshape( [nch*2, 1] )(H)
        H = UpSampling1D(size=4)(H)
        H = Conv1D(int(nch/2), 3, padding='same', kernel_initializer='glorot_uniform')(H)
        H = BatchNormalization()(H)
        H = Activation('relu')(H)
        H = Conv1D(int(nch/4), 3, padding='same', kernel_initializer='glorot_uniform')(H)
        H = BatchNormalization()(H)
        H = Activation('relu')(H)
        H = Conv1D(1, 1, padding='same', kernel_initializer='glorot_uniform')(H)
        g_V = Activation('sigmoid')(H)
        generator = Model(g_input,g_V, name="Generator_model")
        generator.compile(loss='binary_crossentropy', optimizer=self.dopt)
        generator.summary()
        return generator, g_input
    

