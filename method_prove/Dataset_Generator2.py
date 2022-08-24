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

# for x in range(150):
#    y = f(x, 50)
#    xs.append(x)
#    ys.append(y)
#
#plt.plot(xs, ys)
# plt.show()


def Dataset(n_batch, batch_size, exemplos_por_batch):
    XC = []
    out1 = []
    inp1 = []
    out2 = []
    inp2 = []
    T = np.linspace(0, exemplos_por_batch, num=exemplos_por_batch)
    B = np.linspace(0.02, 0.05, num=exemplos_por_batch)
    for _ in range(n_batch):
        t1 = []
        t2 = []
        position1 = []
        position2 = []
        full = 0
        while full != batch_size:
            xc = rd.randint(1, exemplos_por_batch-1)
            BB = rd.randint(0, exemplos_por_batch-1)
            b = B[BB]
            XC.append(xc)
            y1 = []
            y2 = []
            tpred1 = []
            tpred2 = []
            for l in T:
                yy = f(l, b, xc)
                if l < xc:
                    y1.append(yy)
                    tpred1.append(l)
                elif l >= xc:
                    y2.append(yy)
                    tpred2.append(l)
            # plt.clf()
            #plt.xlim([0, exemplos_por_batch])
            #plt.ylim([-1, 1])
            #plt.plot(tpred, y)
            # plt.pause(0.5)
            t1.append(tpred1)
            t2.append(tpred2)
            position1.append(y1)
            position2.append(y2)
            full += 1
        out1.append(position1)
        out2.append(position2)
        inp1.append(t1)
        inp2.append(t2)
        # question.append(t)
    # plt.show()
    XC = np.array(XC).reshape(n_batch, batch_size, 1)
    #inp = np.array(inp).reshape(n_batch, batch_size, exemplos_por_batch)
    inp1 = torch.as_tensor(inp1)
    inp2 = torch.as_tensor(inp2)
    out1 = torch.as_tensor(out1)
    out2 = torch.as_tensor(out2)
    #question = torch.as_tensor(question)
    #print('shape(question) =', np.shape(question))
    print('XC =', np.shape(XC))
    n_batch1 = np.shape(inp1)[0]
    batch_size1 = np.shape(inp1)[1]
    interval1 = np.shape(inp1)[2]
    n_batch2 = np.shape(inp2)[0]
    batch_size2 = np.shape(inp2)[1]
    interval2 = np.shape(inp2)[2]
    print('inp1 =', np.shape(inp1), '{} conjunto(s)'.format(n_batch1) +
          ' de {} exemplos'.format(batch_size1)+', cada um com {} pontos '.format(interval1))
    print('inp2 =', np.shape(inp2), '{} conjunto(s)'.format(n_batch2) +
          ' de {} exemplos'.format(batch_size2)+', cada um com {} pontos '.format(interval2))

    print('out1 =', np.shape(out1), '{} conjunto(s)'.format(n_batch1) +
          ' de {} exemplos'.format(batch_size1)+', cada um com {} pontos '.format(interval1))
    print('out2 =', np.shape(out2), '{} conjunto(s)'.format(n_batch2) +
          ' de {} exemplos'.format(batch_size2)+', cada um com {} pontos '.format(interval2))
    # sys.exit()
    address = open("inp1", "wb")
    pickle.dump(inp1, address)
    address.close()
    address = open("inp2", "wb")
    pickle.dump(inp2, address)
    address.close()
    address = open("out1", "wb")
    pickle.dump(out1, address)
    address.close()
    address = open("out2", "wb")
    pickle.dump(out2, address)
    address.close()
    address = open("XC", "wb")
    pickle.dump(XC, address)
    address.close()
# Dataset(5,10,100)
