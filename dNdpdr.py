import matplotlib.pyplot as plt
import numpy as np
import math
import random
from scipy.optimize import curve_fit
import seaborn as sns
import pandas as pd
from matplotlib import gridspec

def file_open(file):
    import sys
    data = []
    try:
        f = open(file, 'r', encoding='utf-8')
    except Exception:
        print("open error. not found file:", str(file))
        sys.exit(1)
    for line in f:
        line = line.strip() #前後空白削除
        line = line.replace('\n','') #末尾の\nの削除
        line = line.split(",") #分割
        data.append(line)
    f.close()
    return data

data=file_open("../p_rad_RES_2_0xbr_0xshift.txt")
print(data[400])

momentum=[]
radius=[]
n_phi=[]

for i in range(0,64159):
    print("i=",i)
    if len(data[i])==3:
        momentum.append(data[i][0])
        radius.append(data[i][1])
        n_phi.append(data[i][2])

npmomentum=np.array(momentum,dtype=float)
npradius=np.array(radius,dtype=float)
npn_phi=np.array(n_phi,dtype=float)
### 3dグラフを作成

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(npmomentum, npradius, npn_phi, color='blue',s=1,alpha=0.3)
ax.set_xlabel('momentum')
ax.set_ylabel('radius')
ax.set_zlabel('n_phi')

plt.show()