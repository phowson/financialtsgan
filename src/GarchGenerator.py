from arch import arch_model
from CsvDataImport import loadCsv;
import numpy as np;
import math;
import matplotlib.pyplot as plt


class GarchRandomGenerator:
    
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
        
        
    
    

tsList= loadCsv('../data/GBPUSD.csv');

ts = np.array([x for y,x in tsList]);    


retG = [];
lx=ts[0];
for x in ts[1:]:
    retG.append(100 * (x / lx));
    lx = x;
    
retG= np.array(retG);

print(retG)
garch11 = arch_model(retG, p=1, q=1)
res = garch11.fit()
print(res.summary());

# print(res.params["alpha[1]"])
gr = GarchRandomGenerator(alpha = res.params['alpha[1]'], 
                          beta= res.params['beta[1]'], 
                          omega = res.params['omega'], 
                          mu = res.params['mu']);

testRandomData = []
for i in range(0,4000):
    testRandomData.append(gr.next());

plt.plot(np.array(testRandomData));
plt.figure();
plt.plot(retG);
plt.show();
