import os
import io
import numpy as np
import matplotlib.pyplot as plt
import random
import time

path=os.getcwd()+'/data/artData.txt'
data=open(path).read()
data=np.loadtxt(io.StringIO(u''+data))
xData=np.array(data[:,0]);
yData=np.array(data[:,1]);

xTrain=xData#[0:int(.5*len(xData))]
yTrain=yData#[0:int(.5*len(xData))]
trainData=zip(xTrain,yTrain)
trainRand=[random.choice(trainData) for i in range(int(len(xTrain)/10))]
xTrainRand=[trainRand[i][0] for i in range(len(trainRand))]
yTrainRand=[trainRand[i][1] for i in range(len(trainRand))]

xTrain=xTrainRand
yTrain=yTrainRand

  
kernels=[[['sin',5],['Lorentz',4]],[['log',3]],[['ampFreq'],['sin',.5],['x',1]]]
    
    
    
def kerNameToValue(kernel,k,x):
    a=1
    if kernel[k][0]=='x':
        a=x**kernel[k][1]
    elif kernel[k][0]=='c':
        a=1
    elif kernel[k][0]=='exp':
        a=np.exp(kernel[k][1]*x)
    elif kernel[k][0]=='log':
        a=np.log(kernel[k][1]*x)
    elif kernel[k][0]=='log2':
        a=np.log(kernel[k][1]*x)**2
    elif kernel[k][0]=='log3':
        a=np.log(kernel[k][1]*x)**3
    elif kernel[k][0]=='log4':
        a=np.log(kernel[k][1]*x)**4
    elif kernel[k][0]=='log1p':
        a=np.log1p(kernel[k][1]*x)
    elif kernel[k][0]=='cos':
        a=np.cos(kernel[k][1]*x)
    elif kernel[k][0]=='cos2':
        a=np.cos(kernel[k][1]*x)**2
    elif kernel[k][0]=='cos3':
        a=np.cos(kernel[k][1]*x)**3
    elif kernel[k][0]=='cos4':
        a=np.cos(kernel[k][1]*x)**4
    elif kernel[k][0]=='cosh':
        a=np.cosh(kernel[k][1]*x)
    elif kernel[k][0]=='cosh2':
        a=np.cosh(kernel[k][1]*x)**2
    elif kernel[k][0]=='sin':
        a=np.sin(kernel[k][1]*x)
    elif kernel[k][0]=='sin2':
        a=np.sin(kernel[k][1]*x)**2
    elif kernel[k][0]=='sin3':
        a=np.sin(kernel[k][1]*x)**3
    elif kernel[k][0]=='sin4':
        a=np.sin(kernel[k][1]*x)**4 
    elif kernel[k][0]=='sinh':
        a=np.sinh(kernel[k][1]*x)
    elif kernel[k][0]=='sinh2':
        a=np.sinh(kernel[k][1]*x)**2
    elif kernel[k][0]=='tanh':
        a=np.tanh(kernel[k][1]*x)
    elif kernel[k][0]=='tanh2':
        a=np.tanh(kernel[k][1]*x)**2
    elif kernel[k][0]=='Lorentz':
        a=1/(1+(x-kernel[k][1])**2)
    return a
    
    
    
    
def kerVal(x,kernel):
    
    if kernel[0]=='ampFreq':
        phase=random.random()*np.pi
        amp=kerNameToValue(kernel,1,x)
        freq=kerNameToValue(kernel,2,x)
        a=amp*np.sin(freq*x+phase)
    else:
        a=1
        for k in range(len(kernel)):
            a*=kerNameToValue(kernel,k,x)
    return a

    
    
    
def training(student,xTrain,yTrain):
    kerMat=np.zeros((len(xTrain),len(student)))
    i=0
    for x in xTrain:
        j=0
        for kernel in student:
            kerMat[i,j]=kerVal(x,kernel)    
            j+=1 
        i+=1  
    kerMatInv=np.linalg.pinv(kerMat)
    coefs=np.dot(kerMatInv,yTrain)    
    return coefs
        

   


def prediction(student,coefs,xData):
        pred=np.zeros(len(xData))
        i=0
        for x in xData:
            j=0
            for kernel in student:
                pred[i]+=coefs[j]*kerVal(x,kernel)
                j+=1   
            i+=1
        return pred
       
def plot(xContinuous,predict):
        plt.plot(xContinuous,predict)
        plt.plot(xTrain,yTrain,'r.')
        #plt.plot(self.xValid,self.yValid,'g.')
        #plt.plot(self.xTest,self.yTest,'k.')
        plt.show()
        #plt.plot(xContinuous,predict)
        #plt.show()

start=time.clock()
coefs=training(kernels,xTrain,yTrain)
print (coefs)
xContinuous=np.linspace(xData[0],xData[-1],500)
predict=prediction(kernels,coefs,xContinuous) 
print('time=',time.clock()-start)
pl=plot(xContinuous,predict)        

