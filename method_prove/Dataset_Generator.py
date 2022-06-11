import random as rd;import numpy as np;import sys;import pickle
import torch
import torch.nn as nn
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch.optim as optim
import matplotlib.pyplot as plt
import math
#%matplotlib inline


def DampedPend(b,k,t,m):
    if t == 0:
        position = 1
    else:
        dump = math.exp(-(b/2*m)*t)
        omega = np.sqrt(k/m)*np.sqrt(1-(b**2)/(4*m*k))
        osc = np.cos(omega*t)
        position = dump*osc
    return position
position = DampedPend(1,1,1,1)
Y=[]
for i in range(0,100):
    y = DampedPend(0.1,5,i,1)
    Y.append(y)
X = np.linspace(0,len(Y),len(Y))
plt.plot(X,Y)
plt.show()
print(position)

''' preciso pensar em outra função para montar o Dataset'''
''' pensar no shape do output, dependendo da lógica que vamos abordar'''
''' o output pode ser um intervalo e a IA tem que dizer o qu'''



def Dataset(n_batch,batch_size,exemplos_por_batch):
    inp=[];    question=[];    m=1
    #T=[i for i in range(0,50)]
    T=np.linspace(0, 50, num=500)
    K=np.linspace(5, 11, num=50)
    B=np.linspace(0.5, 1.1, num=50)
    KK=[];    BB=[]
#    K=np.linspace(5, 11, num=100)          #those are default values
#    B=np.linspace(0.5,1.1, num=100)        #those are default values
#'''         THIS IS FOR A RANDOM CONFIG OF K AND B'''
    for i in range(n_batch):
        t=[];        position=[];        full=0
        while full!=batch_size:
            ki=rd.randint(0,49);        bi=rd.randint(0,49)
            k=K[ki];        b=B[bi]
            KK.append(k);           BB.append(b)
            y=[];            tpred=[]
            for l in T:
                yy=DampedPend(b,k,l,m)
                y.append(yy)
                tpred.append(l)
            plt.clf()                   #uncoment to graph
            plt.xlim([0, 12])           #uncoment to graph
            plt.ylim([-1, 1])           #uncoment to graph
            plt.plot(tpred,y)           #uncoment to graph
            plt.pause(0.5)              #uncoment to graph

            t.append(tpred)
            position.append(y)
            full+=1
        inp.append(position)
        question.append(t)
    KK=np.array(KK).reshape(n_batch,batch_size,1)   # To works on scynet
    BB=np.array(BB).reshape(n_batch,batch_size,1)   # To works on scynet
    Constantes=[KK,BB]
    inp=torch.as_tensor(inp)
    question=torch.as_tensor(question)
    plt.show()
    print('shape(question) =',np.shape(question))
    print('Constantes =',np.shape(Constantes))
    sys.exit()
    address = open("positions","wb")
    pickle.dump(inp, address)
    address.close()
    address = open("question","wb")
    pickle.dump(question, address)
    address.close()
    address = open("Constantes","wb")
    pickle.dump(Constantes, address)
    address.close()
Dataset(5,1000,50)