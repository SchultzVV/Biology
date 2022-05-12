import matplotlib
matplotlib.use('tkagg')
import glob;import os;import cv2;import matplotlib.image as mpimg
import numpy as np;import pickle;import sys as s 
import matplotlib.pyplot as plt
from pandas_ods_reader import read_ods


sheet = "sheet_name"
df = read_ods('data.ods', headers=True)# , sheet)
sheets=df.keys()

cells=[]
for i in sheets:
    cells.append(i)
cells.remove('time')

#print(df[cells[0]])
#print(df[cells[0]][0])


for i in df[cells]:

    x=df[cells[0]]
y=np.linspace(0,len(x),len(x))    

plt.plot(y,x)
plt.show()
