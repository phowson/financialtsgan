import numpy as np;
import keras;

def createPriceSeriesFromReturns(l, initial, ticksize):
    
    p = initial;
    out = [ p ];
    for x in l:
        p = p * (x /100.);        
        p = round(p/ ticksize) * ticksize;
        out.append(p)
    
    return np.array(out);
    

def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val


class LossHistory(keras.callbacks.Callback):
    
    def __init__(self, model, filename = 'model.keras'):
        self.model = model;
        self.minLoss = 9999e999;
        self.filename = filename;
    

    def on_epoch_end(self, batch, logs={}):
        l = logs.get('loss');        
        print("Loss:")
        print(l)
        
        if (l<self.minLoss):
            print("Saving new best model");
            self.model.save(self.filename)
            self.minLoss = l;