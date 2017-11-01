'''
Created on 31 Oct 2017

@author: phil
'''
import math
import numpy as np;
from CsvDataImport import loadCsv;
import matplotlib.pyplot as plt


class WhiteNoiseModel:
    
    
    def fit(self, ts):
        x= np.array(ts);
        self.mean = np.mean(x);
        self.sd = math.sqrt(np.var(x)) 
        self.minTickSize= 99e999;
        lx=ts[0];        
        for x in ts[1:]:
            sz = abs(lx - x);
            if sz<self.minTickSize and sz>0:
                self.minTickSize = sz;
            lx = x;             
    
    def generate(self, numPoints=1000):
        testRandomData = []
        for i in range(0,numPoints):
            p = self.mean + self.sd * np.random.normal();
            p = round(p/ self.minTickSize) * self.minTickSize;
            testRandomData.append(p);
                
        return np.array(testRandomData);
            

            
def testWhiteNoise():
    tsList= loadCsv('../data/GBPUSD.csv');
    
    ts = np.array([x for y,x in tsList]);    
    
    randomGenerator = WhiteNoiseModel();
    randomGenerator.fit(ts);
        
    plt.figure("Random from trained brownian");
    plt.plot(randomGenerator.generate(8000));
    plt.figure("Original training set");
    plt.plot(ts);
    plt.show();
    
    
#testWhiteNoise()
