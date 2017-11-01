from arch import arch_model
from CsvDataImport import loadCsv;
import numpy as np;
import math;
import matplotlib.pyplot as plt

from Helpers import createPriceSeriesFromReturns;



class Garch11RandomVariable:
    
    def __init__(self, alpha, beta, omega, mu):
        self.et = 0;
        self.alpha = alpha;
        self.beta = beta;
        self.omega = omega;
        self.mu = mu;
        
        self.sigmaSquaredT = omega;
        
    def standardNormalRandom(self):
        return np.random.normal();
        
    def reset(self):
        self.sigmaSquaredT = self.omega;
        self.et = 0;
        
    def next(self):
        
        newSigmaSquaredT = self.omega + self.alpha * self.et* self.et + self.beta*self.sigmaSquaredT;
        self.et = math.sqrt(newSigmaSquaredT) * self.standardNormalRandom();
        rt = self.mu + self.et;
        self.sigmaSquaredT = newSigmaSquaredT;
        return rt;
        
        
class Garch11Model:
    
    def __init__(self):
        self.gr=None;
        self.ts = None;
    
    def fit(self, ts):
        retG = [];
        self.ts = ts;
        lx=ts[0];
        self.minTickSize= 99e999;
        
        for x in ts[1:]:
            retG.append(100 * (x / lx));
            sz = abs(lx - x);
            if sz<self.minTickSize and sz>0:
                self.minTickSize = sz;
            lx = x;            
            
            
            
        retG= np.array(retG);
        garch11 = arch_model(retG, p=1, q=1)
        res = garch11.fit()
        
        self.gr = Garch11RandomVariable(alpha = res.params['alpha[1]'], 
                                  beta= res.params['beta[1]'], 
                                  omega = res.params['omega'], 
                                  mu = res.params['mu']);


        
    def generate(self, numPoints=1000):
        testRandomData = []
        
        # Re-randomise garch
        self.gr.reset()
        for i in range(0,1000):
            self.gr.next();
            
        for i in range(0,numPoints-1):
            testRandomData.append(self.gr.next());
                
        return createPriceSeriesFromReturns(testRandomData, self.ts[int(np.random.uniform() * len(self.ts))], self.minTickSize);
        


def testGarchGenerator():
    
    tsList= loadCsv('../data/GBPUSD.csv');
    
    ts = np.array([x for y,x in tsList]);    
    
    randomGenerator = Garch11Model();
    randomGenerator.fit(ts);
        
    plt.figure("Random from trained garch");
    plt.plot(randomGenerator.generate(8000));
    plt.figure("Original training set");
    plt.plot(ts);
    plt.show();
