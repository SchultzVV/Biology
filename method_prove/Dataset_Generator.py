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


def f(x, xc):
    if x < xc:
        y = math.sin(x)
    else:
        y = math.sin(x)*math.exp(-0.025*x)
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
    inp = []
    question = []
    XC = []
    T = np.linspace(0, exemplos_por_batch, num=exemplos_por_batch)
    for _ in range(n_batch):
        t = []
        position = []
        full = 0
        while full != batch_size:
            xc = rd.randint(5, exemplos_por_batch)
            XC.append(xc)
            y = []
            tpred = []
            for l in T:
                yy = f(l, xc)
                y.append(yy)
                tpred.append(l)
            plt.clf()
            plt.xlim([0, exemplos_por_batch])
            plt.ylim([-1, 1])
            plt.plot(tpred, y)
            plt.pause(0.5)
            t.append(tpred)
            position.append(y)
            full += 1
        inp.append(position)
        question.append(t)
    KK = np.array(KK).reshape(n_batch, batch_size, 1)   # To works on scynet
    BB = np.array(BB).reshape(n_batch, batch_size, 1)   # To works on scynet
    Constantes = [KK, BB]
    inp = torch.as_tensor(inp)
    question = torch.as_tensor(question)
    plt.show()
    print('shape(question) =', np.shape(question))
    print('Constantes =', np.shape(Constantes))
    sys.exit()
    address = open("positions", "wb")
    pickle.dump(inp, address)
    address.close()
    address = open("question", "wb")
    pickle.dump(question, address)
    address.close()
    address = open("Constantes", "wb")
    pickle.dump(Constantes, address)
    address.close()
Dataset(5,1000,50)
