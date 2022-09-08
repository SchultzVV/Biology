import random as rd
import torch
import random as rd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error


def train(model,epochs,inp,out,XC):
    n = np.shape(inp)[0]
    n_b = np.shape(inp)[1]
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    for epoch in range(epochs):
        rnd_n = rd.randint(0,n-1)
        rnd_b = rd.randint(0,n_b-1)
        xc = XC[rnd_n][rnd_b]
        x = inp[rnd_n][rnd_b]
        y = out[rnd_n][rnd_b]
        #x = inp[0][0]
        #y = out[0][0]
        x1=[]
        x2=[]
        y1=[]
        y2=[]
        for k in range(50):
            if x[k] <= xc:
                x1.append(x[k])
                y1.append(y[k])
            else:
                x2.append(x[k])
                y2.append(y[k])
        x1 = np.array(x1)
        x1 = torch.tensor(x1)
        x1 = x1.reshape(len(x1),1).float()
        x2 = np.array(x2)
        x2 = torch.tensor(x2)
        x2 = x2.reshape(len(x2),1).float()
        y1 = np.array(y1)
        y1 = torch.tensor(y1)
        y1 = y1.reshape(len(y1),1).float()
        y2 = np.array(y2)
        y2 = torch.tensor(y2)
        y2 = y2.reshape(len(y2),1).float()
        predict1 = model(x1,1)
        loss = torch.sum((predict1-y1)**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss.item())

        predict2 = model(x2,2)
        loss = torch.sum((predict2-y2)**2 )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        #print(loss.item())
    return model

def get_metrics(model,inp,out,XC):
    n = np.shape(inp)[0]
    n_b = np.shape(inp)[1]
    total_MAE_1 = 0.0
    total_MAE_2 = 0.0
    for i in range (n):
        for j in range(n_b):
            x = inp[i][j]
            y = out[i][j]
            xc = XC[i][j]
            x1=[]
            x2=[]
            y1=[]
            y2=[]
            for k in range(50):
                if x[k] <= xc:
                    x1.append(x[k])
                    y1.append(y[k])
                else:
                    x2.append(x[k])
                    y2.append(y[k])
            x1 = np.array(x1)
            x1 = torch.tensor(x1)
            x1 = x1.reshape(len(x1),1).float()
            x2 = np.array(x2)
            x2 = torch.tensor(x2)
            x2 = x2.reshape(len(x2),1).float()
            y1 = np.array(y1)
            y1 = torch.tensor(y1)
            y1 = y1.reshape(len(y1),1).float()
            y2 = np.array(y2)
            y2 = torch.tensor(y2)
            y2 = y2.reshape(len(y2),1).float()
            predict1 = model(x1,1).detach().numpy()
            predict2 = model(x2,2).detach().numpy()

            y1 = y1.detach().numpy().reshape(1,len(y1))
            y2 = y2.detach().numpy().reshape(1,len(y2))
            predict1 = predict1.reshape(1,len(predict1))
            predict2 = predict2.reshape(1,len(predict2))

            #MSE_1 = mean_squared_error(predict1, y1)
            #RMSE_1 = mean_squared_error(predict1, y1,squared=False)
            MAE_1 = mean_absolute_error(predict1, y1)
            total_MAE_1+=MAE_1

            #MSE_2 = mean_squared_error(predict2, y2)
            #RMSE_2 = mean_squared_error(predict2, y2,squared=False)
            MAE_2 = mean_absolute_error(predict2, y2)
            total_MAE_2+=MAE_2
    MAE_model1 = total_MAE_1/(n*n_b)
    MAE_model2 = total_MAE_2/(n*n_b)
    return MAE_model1, MAE_model2

def get_metrics_one_shot(model,inp,out,xc):
    x = inp[0][0]
    y = out[0][0]
    x1=[]
    x2=[]
    y1=[]
    y2=[]
    for k in range(50):
        if x[k] <= xc:
            x1.append(x[k])
            y1.append(y[k])
        else:
            x2.append(x[k])
            y2.append(y[k])
    x1 = np.array(x1)
    x1 = torch.tensor(x1)
    x1 = x1.reshape(len(x1),1).float()
    x2 = np.array(x2)
    x2 = torch.tensor(x2)
    x2 = x2.reshape(len(x2),1).float()
    y1 = np.array(y1)
    y1 = torch.tensor(y1)
    y1 = y1.reshape(len(y1),1).float()
    y2 = np.array(y2)
    y2 = torch.tensor(y2)
    y2 = y2.reshape(len(y2),1).float()
    predict1 = model(x1,1).detach().numpy()
    predict2 = model(x2,2).detach().numpy()
    y1 = y1.detach().numpy().reshape(1,len(y1))
    y2 = y2.detach().numpy().reshape(1,len(y2))
    predict1 = predict1.reshape(1,len(predict1))
    predict2 = predict2.reshape(1,len(predict2))
    #MSE_1 = mean_squared_error(predict1, y1)
    #RMSE_1 = mean_squared_error(predict1, y1,squared=False)
    MAE_1 = mean_absolute_error(predict1, y1)
    #MSE_2 = mean_squared_error(predict2, y2)
    #RMSE_2 = mean_squared_error(predict2, y2,squared=False)
    MAE_2 = mean_absolute_error(predict2, y2)
    return MAE_1, MAE_2