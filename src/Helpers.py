import numpy as np;

def createPriceSeriesFromReturns(l, initial):
    
    p = initial;
    out = [ p ];
    for x in l:
        p = p * (x /100.);
        
        out.append(p)
    
    return np.array(out);
    

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val

