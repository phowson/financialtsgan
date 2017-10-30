'''
Created on 30 Oct 2017

@author: phil
'''

import Descriminator
import Generator


def make_trainable(net, val):
    net.trainable = val
    for l in net.layers:
        l.trainable = val





