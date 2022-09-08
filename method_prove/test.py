import random as rd
import pickle
import random as rd
import numpy as np
import matplotlib.pyplot as plt
from Dataset_Generator import Dataset
from Linear_model import Linear
from utils import get_metrics, train, get_metrics_one_shot

Dataset(10,20,50)
inp = pickle.load(open("inp", "rb"))
out = pickle.load(open("out", "rb"))
XC = pickle.load(open("XC", "rb"))
print(XC[0][0])
n = np.shape(inp)[0]
n_b = np.shape(inp)[1]


from utils import get_metrics, train
X = []
Y = []
Y1 = []
Y2 = []
for xc in range(15,35):
    model = Linear()
    trained_model = train(model,20,inp,out,XC)
    metric_1, metric_2 = get_metrics(trained_model,inp,out,XC)
    print('xc = ',xc)
    print('metric_1 = ',metric_1)
    print('metric_2 = ',metric_2)
    print('metric_1-metric_2 = ',abs(metric_1-metric_2))
    X.append(xc)
    Y.append(abs(metric_1-metric_2))
    Y1.append(metric_1)
    Y2.append(metric_2)
#print('O valor de divisão xc é {}.'.format(XC[0][0]))
plt.plot(X,Y)
plt.plot(X,Y1,label='mae_model_1')
plt.plot(X,Y2,label='mae_model_2')
plt.legend()
plt.show()
plt.plot(inp[0][0],out[0][0])
plt.show()