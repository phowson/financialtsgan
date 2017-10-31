'''
Created on 30 Oct 2017

@author: phil
'''

import Descriminator
import Generator
import numpy as np;
from  BrownianGenerator import BrownianModel 
from CsvDataImport import loadCsv;
from keras.optimizers import Adam



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




randomGenerator = BrownianModel();
randomGenerator.fit(ts);

for strideX in range(numRealSamples, numRealSamples+numFakeSamples ):
    x[strideX] = randomGenerator.generate(windowSize)
    y[strideX][1] = 1; 


x=np.reshape(x, (numRealSamples+numFakeSamples, windowSize, 1))


print("Created training set, with shape")
print(x.shape);



hist = descrModel.fit(x, y,  
              batch_size=256, epochs=10, verbose=1)


yPrime = descrModel.predict(x);



for strideX in range(0, numRealSamples+numFakeSamples ):
    print(yPrime[strideX])


