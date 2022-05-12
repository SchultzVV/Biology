import glob;import os;import matplotlib.image as mpimg;import matplotlib.pyplot as plt
import numpy as np;import pickle;import sys as s;import random as rd
import torch;import torch.nn as nn;import torch.optim as optim
from pandas_ods_reader import read_ods



#---------------------------------------------------------------------------------------
def Get_Dataset_math(n_b,b_idx,b_size):# Esse é o nome da função.
    Size=n_b*b_idx
    print(Size)
    O=[];    Q=[];    A=[]
    for i in range(Size):
        a=rd.randint(0,200);        b=rd.randint(200,200+b_size)
        X=np.linspace(a,b,int(b_size/2));        x=np.linspace(a,b,b_size)
        y=x2(X);        y=y/max(y);        Y=seno(X)
#        print(y);        print(Y)
        YY=np.concatenate((y,Y), axis=0)
        O.append(YY)
        r=rd.randint(1,len(x)-1)
        Q.append(x[r]);        A.append(YY[r])

    
    O=torch.as_tensor(O)
    A=torch.as_tensor(A)
    Q=torch.as_tensor(Q)
    O=O.reshape(n_b,b_idx,b_size)
    Q=Q.reshape(n_b,b_idx,1)
    A=A.reshape(n_b,b_idx,1)
    print(np.shape(O));        print(np.shape(Q));        print(np.shape(A))
    return O,Q,A
        #plt.plot(x,YY)
        #plt.show()
        #s.exit()
#---------------------------------------------------------------------------------------
# roda a função que foi escrita agora  Get_Dataset_math(), ela retorna 3 objetos.
#O,Q,A=Get_Dataset_math(100,10,50)
#n_batch=np.shape(O)[0]
#batch_size=np.shape(O)[1]
#n=np.shape(O)[2]
#---------------------------------------------------------------------------------------
def Get_Dataset_real():# Esse é o nome da função.
    sheet = "sheet_name"
    df = read_ods('data_without_low_data.ods', headers=True)
    sheets=df.keys()
    cells=[];    Y=[];    X=[]
    for i in sheets:
        cells.append(i)
    cells.remove('time')
    o=1
    for i in cells:
        o+=1
        y=np.array(df[i])
        y=y/max(y)
        x=np.linspace(0,len(y),len(y))
        Y.append(y)
        X.append(x)
    Y=torch.as_tensor(Y)
    print(np.shape(Y))
    return Y
#---------------------------------------------------------------------------------------

O=Get_Dataset_real()
#y=O[2]
#x=np.linspace(0,len(y),len(y))
#plt.plot(x,y)
#plt.show()
#s.exit()
#---------------------------------------------------------------------------------------
def Get_Dataset_real():# Esse é o nome da função.
    sheet = "sheet_name"
    df = read_ods('data_without_low_data.ods', headers=True)
    sheets=df.keys()
    cells=[];    Y=[];    X=[]
    for i in sheets:
        cells.append(i)
    cells.remove('time')
    o=1
    for i in cells:
        o+=1
        y=np.array(df[i])
        y=y/max(y)
        x=np.linspace(0,len(y),len(y))
        Y.append(y)
        X.append(x)
    Y=torch.as_tensor(Y)
    print(np.shape(Y))

    address = open("Y_train","wb");    pickle.dump(Y, address);    address.close()
    return Y
Y=Get_Dataset_real()


