'''
Created on 31 Oct 2017

@author: phil
'''

import math
import numpy as np;
from CsvDataImport import loadCsv;
import matplotlib.pyplot as plt
from Helpers import createPriceSeriesFromReturns;


class BrownianModel:
    
    
    def fit(self, ts):
        retG = [];
        self.ts = ts;
        lx=ts[0];
        self.minTickSize= 99e999;        
        for x in ts[1:]:
            retG.append(math.log(x / lx));
            sz = abs(lx - x);
            if sz<self.minTickSize and sz>0:
                self.minTickSize = sz;
            lx = x;            
        retG= np.array(retG);
        self.mean = np.mean(retG);
        self.sd = math.sqrt(np.var(retG)) 
       
    
    def generate(self, numPoints=1000):
        testRandomData = []
        for i in range(0,numPoints-1):
            testRandomData.append(100*math.exp(self.mean + self.sd * np.random.normal()));
                
        return createPriceSeriesFromReturns(testRandomData, self.ts[int(np.random.uniform() * len(self.ts))], self.minTickSize);
            

            
def testBrownian():
    tsList= loadCsv('../data/GBPUSD.csv');
    
    ts = np.array([x for y,x in tsList]);    
    
    randomGenerator = BrownianModel();
    randomGenerator.fit(ts);
        
    plt.figure("Random from trained brownian");
    plt.plot(randomGenerator.generate(8000));
    plt.figure("Original training set");
    plt.plot(ts);
    plt.show();
    
    
#testBrownian()
