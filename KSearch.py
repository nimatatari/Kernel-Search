import io
import os
import numpy as np
import problems
import matplotlib.pyplot as plt
import time

#=================================================================================
# data
path=os.getcwd()+"/data/artData.txt"
data=open(path).read()
#data=data.split('\n'))
#xData=[float((data.split('\n')[i]).split(',')[0]) for i in range(200)]
#yData=[float((data.split('\n')[i]).split(',')[1]) for i in range(200)]
data=np.loadtxt(io.StringIO(u''+data))
xData=np.array(data[:,0]);
yData=np.array(data[:,1]);
#=================================================================================
# parameters
trainSetRatio=0.5
validationSetRatio=0.6
poolSize=400
bestClassSize=1
goalDepth=4
replica=4
runs=1
#=================================================================================
# problem solving

predict=[]
xContinuous=np.linspace(xData[0],xData[-1],500)
ti=time.time()
for run in range(runs):
    print ('run=%i/%i'%(run,runs))
    pred=[]
    normalize=0
    for r in range(replica):
        print ('      replica=%i/%i'%(r+1,replica))
        problem=problems.KSearch(xData,yData,trainSetRatio,validationSetRatio,poolSize,replica,bestClassSize,goalDepth)
        result=problem.findBaseFunction()
        w=(1/result[0])**3
        normalize+=w
        pred.append(np.dot(w,result[1]))
    predict.append(np.dot(1/normalize,np.sum(pred,0)))
tf=time.time()
t=(tf-ti)/runs*1.
problem.plot(xContinuous,predict,t)
plt.show()
    
#=================================================================================
