import numpy as np
import pickle
import torch
from pandas_ods_reader import read_ods


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


''' Gera o dataset
        O dataset é composto de $part_1=[y_0,..,y_{d/2}]$ with dimension $d/2$ and $label_1=[0]$ For the initial part of the graph.
        For the second part of the graph we have $part_2=[y_{d/2},...,y_d]$ with dimension $d/2$ and with $label_2=[1]$.
'''


def gen_dataset2():
    Y = Get_Dataset_real()
    Input = []
    Output = []
    Input_test = []
    Output_test = []
    j = 0
    for i in Y:
        if j < 10:
            y1 = i[0:30]
            Input.append(y1)
            Output.append(0)
            y2 = i[len(i)-30:len(i)]
            Input.append(y2)
            Output.append(1)
            j += 1
        else:
            y1 = i[0:30]
            Input_test.append(y1)
            Output_test.append(0)
            y2 = i[len(i)-30:len(i)]
            Input_test.append(y2)
            Output_test.append(1)
    # ---------------------- Salva o dataset com os 80 exemplos, ordenado em 4 grupos de 20
    print('O Input tem tamanho {}'.fotmat(np.shape(Input)))
    print('O Output tem tamanho {}'.fotmat(np.shape(Output)))
    print('O test_Input tem tamanho {}'.fotmat(np.shape(Input_test)))
    print('O test_Output tem tamanho {}'.fotmat(np.shape(Output_test)))
    # s.exit()
    Input = np.reshape(Input, (2, 10, 30))
    Output = np.reshape(Output, (2, 10, 1))
    Input = torch.as_tensor(Input)
    Output = torch.as_tensor(Output)
    address = open("Input_Train", "wb")
    pickle.dump(Input, address)
    address.close()
    address = open("Train_Labels", "wb")
    pickle.dump(Output, address)
    address.close()
    print("Input_Train", np.shape(Input))
    print("Train_Labels", np.shape(Output))

    # ---------------------- Salva o dataset com os 8 exemplos, ordenado em 4 grupos de 2
    Input_test = np.reshape(Input_test, (7, 10, 30))
    Output_test = np.reshape(Output_test, (7, 10, 1))
    Input_test = torch.as_tensor(Input_test)
    Output_test = torch.as_tensor(Output_test)
    address = open("Input_test", "wb")
    pickle.dump(Input_test, address)
    address.close()
    address = open("test_Labels", "wb")
    pickle.dump(Output_test, address)
    address.close()
    print("Input_test", np.shape(Input_test))
    print("test_Labels", np.shape(Output_test))
    return Input, Output, Input_test, Output_test


#Input, Output, Input_test, Output_test = gen_dataset()

def gen_dataset(n_batch):
    Y = Get_Dataset_real()
    Input = []
    Output = []
    Input_test = []
    Output_test = []
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
            Input_test.append(y1)
            Output_test.append(0)
            y2 = i[len(i)-30:len(i)]
            Input_test.append(y2)
            Output_test.append(1)
    # ---------------------- Salva o dataset com os 80 exemplos, ordenado em 4 grupos de 20
    print(np.shape(Input))
    print(np.shape(Output))
    print(np.shape(Input_test))
    print(np.shape(Output_test))
    # s.exit()
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
    print("Input_Train", np.shape(Input))
    print("Train_Labels", np.shape(Output))

    # ---------------------- Salva o dataset com os 8 exemplos, ordenado em 4 grupos de 2
    Input_test = np.reshape(Input_test, (abs(9-n_batch), 10, 30))
    Output_test = np.reshape(Output_test, (abs(9-n_batch), 10, 1))
    Input_test = torch.as_tensor(Input_test)
    Output_test = torch.as_tensor(Output_test)
    address = open("Input_test", "wb")
    pickle.dump(Input_test, address)
    address.close()
    address = open("test_Labels", "wb")
    pickle.dump(Output_test, address)
    address.close()
    print("Input_test", np.shape(Input_test))
    print("test_Labels", np.shape(Output_test))
    return Input, Output, Input_test, Output_test


#Input, Output, Input_test, Output_test = gen_dataset(2)
