'''
Created on 30 Oct 2017

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


class LossHistory(keras.callbacks.Callback):
    
    def __init__(self, model):
        self.model = model;
        self.minLoss = 9999e999;
    

    def on_epoch_end(self, batch, logs={}):
        l = logs.get('loss');        
        if (l<self.minLoss):
            print("Saving new best model");
            self.model.save("model.keras")
            self.minLoss = l;        
        

def rolling_window(arr, window):
    """Very basic multi dimensional rolling window. window should be the shape of
    of the desired subarrays. Window is either a scalar or a tuple of same size
    as `arr.shape`.
    """
    shape = np.array(arr.shape*2)
    strides = np.array(arr.strides*2)
    window = np.asarray(window)
    shape[arr.ndim:] = window # new dimensions size
    shape[:arr.ndim] -= window - 1
    if np.any(shape < 1):
        raise ValueError('window size is too large')
    return np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

windowSize = 1000;

optimizer = Adam(lr=1e-3)
factory = Descriminator.DescriminatorFactory((windowSize,1), 0.25, optimizer)
descrModel = factory.create();


tsList= loadCsv('../data/GBPUSD.csv');
ts = np.array([x for y,x in tsList]);  


windows = rolling_window(ts, windowSize);


numRealSamples = ts.shape[0]-windowSize;
numFakeSamples = numRealSamples 

x = np.zeros((numRealSamples+numFakeSamples, windowSize));
y = np.zeros((numRealSamples+numFakeSamples, 2));

for strideX in range(0, numRealSamples ):
    x[strideX] = windows[strideX][:]
    y[strideX][0] = 1;    



randomGenerators = [];
randomGenerators.append(WhiteNoiseModel());
randomGenerators.append(BrownianModel());
randomGenerators.append(Garch11Model());

for g in randomGenerators:
    g.fit(ts);

print("Generating synthetic series")
for strideX in range(numRealSamples, numRealSamples+numFakeSamples ):
    x[strideX] = randomGenerators[strideX % len(randomGenerators)].generate(windowSize)
    y[strideX][1] = 1; 


x=np.reshape(x, (numRealSamples+numFakeSamples, windowSize, 1))


print("Created training set, with shape")
print(x.shape);



history = LossHistory(descrModel);
descrModel.fit(x, y,  
              batch_size=256, epochs=500, verbose=1, callbacks=[history])


yPrime = descrModel.predict(x);



with open("results.txt","w") as file: 

    for strideX in range(0, numRealSamples+numFakeSamples ):
        file.write(str(yPrime[strideX]))
        file.write('\n');


