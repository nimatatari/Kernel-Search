import io
import os
import numpy as np
import matplotlib.pyplot as plt
#=================================================================================
# data
path=os.getcwd()+"/data/fig1-observed-H.txt"
data=open(path).read()
data=np.loadtxt(io.StringIO(u''+data))
xData=data[:,0]
yData=data[:,1]

plt.plot(xData,yData)
plt.show()
