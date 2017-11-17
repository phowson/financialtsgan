
from keras.utils import np_utils
import keras.models as models
from keras.layers import Input,merge, Concatenate
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
from keras.layers import Layer
from keras import backend as K
from keras.layers import LSTM
import math;



#Custom loss funciton
class NormalPDFLogLikelyhoodLayer(Layer):
    
    def __init__(self, **kwargs):
        self.is_placeholder = True
        self.scalingFactor = 1./math.sqrt(2*math.pi);
        super(NormalPDFLogLikelyhoodLayer, self).__init__(**kwargs)    

    def call(self, inputs):
        mu = inputs[0];
        sigmaSquared = inputs[1]+1e-5;
        observation = inputs[2];
   
        
        
        pdfValue = self.scalingFactor  * K.exp(-K.square(observation - mu) / (2 * sigmaSquared) ) / K.sqrt(sigmaSquared)
        
        
        logLikelyhood = K.log(pdfValue)
        # Having some issues with stability
        loss = K.exp(-K.mean(logLikelyhood));
        
        
        #logLikelyhood = K.mean(K.log(sigmaSquared)-K.square(observation - mu)/ (2*sigmaSquared));
        #loss = logLikelyhood;
        self.add_loss(loss, inputs=inputs)
        
        # Output is not relevant
        return inputs;

class GeneratorFactory:
    def __init__(self, dopt, shp=[25], dropout_rate = 0.25):
        self.inputShape = shp;
        self.dopt = dopt; 
        self.dropout_rate = dropout_rate;
        
    def createLSTM(self, look_back = 10, batch_size = 128):
        g_input = Input(shape=self.inputShape, batch_shape = (batch_size,look_back, 1))
        H = g_input
        H = LSTM(8, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True)(H);
        H = LSTM(8, batch_input_shape=(batch_size, look_back, 1), stateful=True)(H);
        H = Dense(8)(H)
        H = LeakyReLU(0.1)(H)
        H = Dense(4)(H)
        H = LeakyReLU(0.1)(H)
        LowLevelMean = Dense(1, activation = "sigmoid")(H);
        
        
        H2 = g_input
        H2 = LSTM(8, batch_input_shape=(batch_size, look_back, 1), stateful=True, return_sequences=True)(H2);
        H2 = LSTM(8, batch_input_shape=(batch_size, look_back, 1), stateful=True)(H2);
        H2 = Dense(8)(H2)
        H2 = LeakyReLU(0.1)(H2)
        H2 = Dense(4)(H2)
        H2 = LeakyReLU(0.1)(H2)
        LowLevelVar = Dense(1, activation = "sigmoid")(H2);
        
        H2 = Concatenate()([H2, LowLevelMean]);
        
        g_V2 = Dense(1,activation='sigmoid', name="PredictedVariance")(H2)


        #

        H = Concatenate()([H, LowLevelVar])
        
        g_V1 = Dense(1,activation='sigmoid', name='PredictedMean')(H)


        generator = Model(g_input,[g_V1, g_V2] , name="Generator_model")
        generator.compile(loss='mse', optimizer=self.dopt)
        generator.summary()
        return generator, g_input
        
    def create(self, nch = 100):
        
        g_input = Input(shape=self.inputShape)
        H = g_input
        H = Conv1D(128,  kernel_size=5, strides=2, dilation_rate=1,  padding = 'same', activation='relu')(H)       
        H = LeakyReLU(0.1)(H)        
        H = Dropout(self.dropout_rate)(H)
        H = Conv1D(32,  kernel_size=3, strides=2, dilation_rate=1, padding = 'same', activation='relu')(H)
        H = LeakyReLU(0.1)(H)
        H = Dropout(self.dropout_rate)(H)        
        H = Flatten()(H)

        H = Dense(16)(H)
        H = LeakyReLU(0.1)(H)
        H = Dense(8)(H)
        H = LeakyReLU(0.1)(H)        
        H = Dropout(self.dropout_rate)(H)
        LowLevelMean = Dense(1)(H);

        H2 = g_input
        H2 = Conv1D(128,  kernel_size=5, strides=2, dilation_rate=1,  padding = 'same', activation='relu')(H2)       
        H2 = LeakyReLU(0.1)(H2)        
        H2 = Dropout(self.dropout_rate)(H2)
        H2 = Conv1D(32,  kernel_size=3, strides=2, dilation_rate=1, padding = 'same', activation='relu')(H2)
        H2 = LeakyReLU(0.1)(H2)
        H2 = Dropout(self.dropout_rate)(H2)        
        H2 = Flatten()(H2)

        H2 = Dense(16)(H2)
        H2 = LeakyReLU(0.1)(H2)
        H2 = Dense(8)(H2)
        H2 = LeakyReLU(0.1)(H2)
        
        H2 = Dropout(self.dropout_rate)(H2)
        LowLevelVar = Dense(1)(H2);
        H2 = Concatenate()([H2, LowLevelMean]);
        
        g_V2 = Dense(1,activation='sigmoid', name="PredictedVariance")(H2)


        #

        H = Concatenate()([H, LowLevelVar])
        
        g_V1 = Dense(1,activation='sigmoid', name='PredictedMean')(H)


        generator = Model(g_input,[g_V1, g_V2] , name="Generator_model")
        generator.compile(loss='mse', optimizer=self.dopt)
        generator.summary()
        return generator, g_input
    
    
    def createLossModel(self, overallInput, generatorOutput, observed):
        
        
        H=NormalPDFLogLikelyhoodLayer()([generatorOutput[0],generatorOutput[1], observed]);
        lossModel = Model([overallInput,observed],H , name="Loss_model")
        lossModel.compile(loss=None, optimizer="rmsprop")
        
        return lossModel;
        