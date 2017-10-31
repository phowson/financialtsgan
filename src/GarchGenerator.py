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
        
    def next(self):
        
        newSigmaSquaredT = self.omega + self.alpha * self.et* self.et + self.beta*self.sigmaSquaredT;
        self.et = math.sqrt(newSigmaSquaredT) * self.standardNormalRandom();
        rt = self.mu + self.et;
        self.sigmaSquaredT = newSigmaSquaredT;
        return rt;
        
        
class Garch11Model:
    
    def __init__(self):
        self.gr=None;
        self.initialValue = 0;
    
    def fit(self, ts):
        retG = [];
        self.initialValue = ts[0];
        lx=ts[0];
        for x in ts[1:]:
            retG.append(100 * (x / lx));
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
        for i in range(0,10000):
            self.gr.next();
            
        for i in range(0,numPoints-1):
            testRandomData.append(self.gr.next());
                
        return createPriceSeriesFromReturns(testRandomData, self.initialValue);
        


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
