import random as rd
import numpy as np
import sys
import pickle
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.optim as optim
import sys as s
# %matplotlib inline

import matplotlib.pyplot as plt
import math


def f(x, b, xc):
    if x < xc:
        y = math.sin(x)
    else:
        y = math.sin(x)*math.exp(-b*x)
    return y


#xs = []
#ys = []

#for x in range(150):
#    y = f(x, 50)
#    xs.append(x)
#    ys.append(y)
#
#plt.plot(xs, ys)
#plt.show()


def Dataset(n_batch, batch_size, exemplos_por_batch):
    out = []
    XC = []
    inp = []
    T = np.linspace(0, exemplos_por_batch, num=exemplos_por_batch)
    B = np.linspace(0.02, 0.05, num=exemplos_por_batch)
    for _ in range(n_batch):
        t = []
        position = []
        full = 0
        while full != batch_size:
            xc = rd.randint(1, exemplos_por_batch-1)
            BB = rd.randint(0, exemplos_por_batch-1)
            b = B[BB]
            XC.append(xc)
            y = []
            tpred = []
            for l in T:
                yy = f(l, b, xc)
                y.append(yy)
                tpred.append(l)
            #plt.clf()
            #plt.xlim([0, exemplos_por_batch])
            #plt.ylim([-1, 1])
            #plt.plot(tpred, y)
            #plt.pause(0.5)
            t.append(tpred)
            position.append(y)
            full += 1
        out.append(position)
        inp.append(t)
        #question.append(t)
    #plt.show()
    XC = np.array(XC).reshape(n_batch, batch_size, 1)   # To works on scynet
    #inp = np.array(inp).reshape(n_batch, batch_size, exemplos_por_batch)
    inp = torch.as_tensor(inp)
    out = torch.as_tensor(out)
    #question = torch.as_tensor(question)
    #print('shape(question) =', np.shape(question))
    print('XC =', np.shape(XC))
    print('inp =', np.shape(inp),'{} conjunto(s)'.format(n_batch)+' de {} exemplos'.format(batch_size)+', cada um com {} pontos '.format(exemplos_por_batch))
    
    print('out =', np.shape(out))
    #sys.exit()
    address = open("inp", "wb")
    pickle.dump(inp, address)
    address.close()
    address = open("out", "wb")
    pickle.dump(out, address)
    address.close()
    address = open("XC", "wb")
    pickle.dump(XC, address)
    address.close()
#Dataset(5,10,100)

