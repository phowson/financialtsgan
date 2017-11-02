'''
Created on 2 Nov 2017

@author: phil
'''
from  BrownianGenerator import BrownianModel 
from WhiteNoiseGenerator import WhiteNoiseModel
from GarchGenerator import Garch11Model
import numpy as np;


class RangeNormaliser:
    def __init__(self, origTs):
        self.minInRealData = np.min(origTs);
        self.maxInRealData = np.max(origTs);
        
        
    def normalise(self, ts):
        ts = np.subtract(ts, self.minInRealData);
        ts = np.true_divide(ts, self.maxInRealData-self.minInRealData);
        return ts;

        
        

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


class TrainingSetGenerator:
    
    def __init__(self, windowSize, numRealSamples, numFakeSamples):
        self.numRealSamples = numRealSamples;
        self.numFakeSamples = numFakeSamples; 
        self.windowSize = windowSize;
        self.randomGenerators = [];
        self.forceCorrectRange = [];
        
        self.randomGenerators.append(WhiteNoiseModel());
        self.forceCorrectRange.append(False);
        
        self.randomGenerators.append(BrownianModel());
        self.forceCorrectRange.append(True);
        
        self.randomGenerators.append(Garch11Model());
        self.forceCorrectRange.append(True);
    
    def create(self, tsList):
        origTs = np.array([x for _,x in tsList]);
        
        normaliser = RangeNormaliser(origTs);
        ts = normaliser.normalise(origTs);
          
        # plt.figure("Normalised input series");
        # plt.plot(ts);
        # plt.show();
        
        
        windows = rolling_window(ts, self.windowSize);
        
        

        
        x = np.zeros((self.numRealSamples+self.numFakeSamples, self.windowSize));
        y = np.zeros((self.numRealSamples+self.numFakeSamples, 2));
        
        for strideX in range(0, self.numRealSamples ):
            x[strideX] = windows[strideX][:]
            y[strideX][0] = 1;    
        
        

        
        for g in self.randomGenerators:
            g.fit(origTs);
        
        
        
        print("Generating synthetic series")
        for strideX in range(self.numRealSamples, self.numRealSamples+self.numFakeSamples ):
            x1 = strideX % len(self.randomGenerators);
            tries = 0;
            while True:
                randomSeries = normaliser.normalise(self.randomGenerators[x1].generate(self.windowSize));
                
        
                
                if not self.forceCorrectRange[x1] or ( np.min(randomSeries)>=0 and np.max(randomSeries)<=1 ):
                    break; 
                #print("Regenerate ranodm series " + str(tries));
                tries = tries +1;
            
        #     plt.figure("Random input series");
        #     plt.plot(randomSeries);
        #     plt.show();
            
            
            x[strideX] = randomSeries;
            y[strideX][1] = 1; 
        
        
        x=np.reshape(x, (self.numRealSamples+self.numFakeSamples, self.windowSize, 1))
        
        
        print("Created training set, with shape")
        print(x.shape);
        return (x,y);
        
    
    
