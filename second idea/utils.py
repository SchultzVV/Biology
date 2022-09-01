import matplotlib.pyplot as plt
import numpy as np
import pickle
import random as rd
import torch
import torch.nn as nn
from pandas_ods_reader import read_ods

from modelo import Linear

def Get_Dataset_real():  # Esse é o nome da função.
    df = read_ods('data2.ods', headers=False)
    colunas = df.columns
    y = []
    Y = []
    for i in colunas:
        for j in df[i]:
            if j != 0.0:
                y.append(j)
            else:
                break
        Y.append(y)
        y = []
    address = open("Y_train", "wb")
    pickle.dump(Y, address)
    address.close()
    return Y

def gen_dataset(n_batch,explanation):
    Y = Get_Dataset_real()
    Input = []
    Output = []
    Input_Test = []
    Output_Test = []
    j = 0
    for i in Y:
        if j < n_batch*5:
            y1 = i[0:30]
            Input.append(y1)
            Output.append(0)
            y2 = i[len(i)-30:len(i)]
            Input.append(y2)
            Output.append(1)
            j += 1
        else:
            y1 = i[0:30]
            Input_Test.append(y1)
            Output_Test.append(0)
            y2 = i[len(i)-30:len(i)]
            Input_Test.append(y2)
            Output_Test.append(1)
    
    # ---------------------- Realiza uma explicação da separação do dataset
    if explanation == True :
        print('Nosso dataset possui {} gráficos que representam o sinal de uma célula até completar a mitose'.format(np.shape(Y)[0]))
        print('Cada gráfico gera dois exemplos de treino, então o nosso dataset tem tamanho {}. '.format(np.shape(Y)[0]*2))
        print('Separando o dataset por um dado de entrada (Input) e um valor correto de saída (Output) :')
        print('np.shape(Input) = ',np.shape(Input),': os {} primeiros gráficos de intervalo'.format(np.shape(Input)[0]),' {}.'.format(np.shape(Input)[1]))
        print('np.shape(Output) = ',np.shape(Output),': o vetor de {} valores corretos, que segundo a organização do dataset, é composto de [0,1,0,1,...]'.format(np.shape(Output)[0]))
        print('np.shape(Input_Test)',np.shape(Input_Test),': os últimos {} gráficos de intervalo'.format(np.shape(Input_Test)[0]),' {}.'.format(np.shape(Input_Test)[1]))
        print('np.shape(Output_Test)',np.shape(Output_Test),': o vetor de {} valores corretos é composto de [0,1,0,1,...]'.format(np.shape(Output_Test)[0]))
        print('Separando o dataset em grupos de 10, ou seja, batch_size = 10. O dataset fica pronto para realizar o treino')
    # ---------------------- Salva o conjunto de treino
    Input = np.reshape(Input, (n_batch, 10, 30))
    Output = np.reshape(Output, (n_batch, 10, 1))
    Input = torch.as_tensor(Input)
    Output = torch.as_tensor(Output)
    address = open("Input_Train", "wb")
    pickle.dump(Input, address)
    address.close()
    address = open("Train_Labels", "wb")
    pickle.dump(Output, address)
    address.close()
    if explanation == True :
        print("Input_Train", np.shape(Input))
        print("Train_Labels", np.shape(Output))

    # ---------------------- Salva o conjunto de teste
    Input_Test = np.reshape(Input_Test, (abs(9-n_batch), 10, 30))
    Output_Test = np.reshape(Output_Test, (abs(9-n_batch), 10, 1))
    Input_Test = torch.as_tensor(Input_Test)
    Output_Test = torch.as_tensor(Output_Test)
    address = open("Input_Test", "wb")
    pickle.dump(Input_Test, address)
    address.close()
    address = open("test_Labels", "wb")
    pickle.dump(Output_Test, address)
    address.close()
    if explanation == True :
        print("Input_Test", np.shape(Input_Test))
        print("test_Labels", np.shape(Output_Test))
        print('Por exemplo, inp[0][0] representa os primeiros 30 pontos e inp[0][1] os últimos 30 pontos do mesmo gráfico')
    return Input, Output, Input_Test, Output_Test


def reset_weight(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.orthogonal_(m.weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5)


def treine(model, epochs):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-4, weight_decay=1e-5)
    inp = pickle.load(open("Input_Train", "rb"))
    out = pickle.load(open("Train_Labels", "rb"))
    n_batch = np.shape(inp)[0]
    for epoch in range(epochs):
        for batch_idx in range(n_batch):
            O = inp[batch_idx]
            A = out[batch_idx]
            O = O.float()
            A = A.float()
            recon = model(O)
            loss = torch.mean((recon-A)**2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        #print(f'Epoch:{epoch+1},Loss:{loss.item():.4f}')


def random_predict(model):
    inp = pickle.load(open("Input_Train", "rb"))
    a = np.shape(inp)[0]-1
    b = np.shape(inp)[1]-1
    a_idx = rd.randint(0, a)
    b_idx = rd.randint(0, b)
    out = pickle.load(open("Train_Labels", "rb"))
    O = inp[a_idx]
    y = 1*O[b_idx]
    x = np.linspace(0, len(y), len(y))
    plt.plot(x, y)
    plt.show()
    A = out[a_idx]
    O = O.float()
    A = A.float()
    resposta = model(O)
    print('resposta = ', resposta[b_idx])
    print('label = ', A[b_idx])


def Eval_metric(model, mode):
    if mode == 'test':
        inp = pickle.load(open("Input_Test", "rb"))
        out = pickle.load(open("test_Labels", "rb"))
        size = np.shape(inp)[0]*np.shape(inp)[1]
        inp = inp.reshape(size, 30)
        out = out.reshape(size, 1)
    elif mode == 'treino':
        inp = pickle.load(open("Input_Train", "rb"))
        out = pickle.load(open("Train_Labels", "rb"))
        size = np.shape(inp)[0]*np.shape(inp)[1]
        inp = inp.reshape(size, 30)
        out = out.reshape(size, 1)
    O = inp.float()
    A = out.float().detach().numpy()
    resposta = model(O).detach().numpy()
    tolerance = 0.1
    erro = 0
    acerto = 0
    for i in range(len(resposta)):
        if A[i] < 0.00001 and resposta[i] < [0.5]:
            acerto += 1
        elif A[i] > 0.9999 and resposta[i] > [0.5]:
            acerto += 1
        else:
            erro += 1
    #print('erro = ', erro/size)
    #print('acerto = ', acerto/size)
    return erro/size, acerto/size




#y1, y2, epochs = eval_model(20, 100)

def predict(model,inp,out,n_batch, batch_idx):
    O = inp[n_batch]
    #y = 1*O[batch_idx]
    #x = np.linspace(0, len(y), len(y))
    #plt.plot(x, y)
    #plt.show()
    A = out[n_batch]
    O = O.float()
    A = A.float()
    resposta = model(O)
    print('---------------------------------------------')
    #print('---------------------------------------------')
    #print('O modelo foi criado para dar como resultado: ')
    print('* 0 quando for a primeira metade do gráfico')
    print('* 1 quando for a segunda metade do gráfico')
    print('Para o exemplo de indice {}, o resultado foi'.format([n_batch,batch_idx]))
    print('resposta do modelo = ', resposta[batch_idx].item())
    print('resposta correta = ', A[batch_idx].item())
    #print('---------------------------------------------')
    print('---------------------------------------------')
