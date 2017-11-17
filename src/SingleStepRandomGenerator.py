'''
Created on 16 Nov 2017

@author: phil
'''
from TrainingSet import TrainingSetGenerator,rolling_window, RangeNormaliser
import numpy as np;
import math;
import matplotlib.pyplot as plt

class SingleStepTSGenerator:
    def __init__(self, model, windowSize, outWindowSize, inputData, batch_size):
        self.model = model;
        normaliser = RangeNormaliser(inputData);
        
        self.windowSize = windowSize;
        self.outWindowSize = outWindowSize;
        self.inputData = normaliser.normalise(inputData);
        self.batch_size = batch_size;
        
        
    def generate(self):
        rw = rolling_window(self.inputData, self.windowSize);
        startx = int(np.random.uniform() * (self.inputData.shape[0] - self.windowSize -self.batch_size-1));
        #print(startx)
        gb = np.zeros((self.batch_size,self.windowSize));
        print("Setup state");
#         for i in range(0, self.batch_size):
#             for k in range(0, self.windowSize):
#                 gb[i][k][0] = rw[startx][k];

              
        for i in range(0, self.batch_size):
            gb[i] = rw[startx][:];
        
        gb = gb.reshape((self.batch_size,self.windowSize,1))
        
        outBuffer = np.zeros((self.outWindowSize));
        self.model.reset_states();
        print("Generate");
        for i in range(0, self.outWindowSize):
            #print(gb.shape)
            p = self.model.predict(gb, batch_size=self.batch_size);
            rs = p[0][self.windowSize-1] + np.random.normal(scale = math.sqrt(p[1][self.windowSize-1])) ;
            outBuffer[i] = rs;
            
            
            for z in range(0, self.batch_size-1):
                gb[z] = gb[z+1];
            
            
            for z in range(0, self.windowSize-1):
                gb[self.batch_size-1][z][0] = gb[self.batch_size-1][z+1][0];
                
            gb[self.batch_size-1][self.windowSize-1][0] = rs;
#             q = np.reshape(gb[self.batch_size-1], (self.windowSize));
#             print(q.shape);
#             plt.plot(q);
#             plt.show();
        return outBuffer;

