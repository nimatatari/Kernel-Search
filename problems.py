# problems.py
import numpy as np
import scipy
from scipy import optimize
import matplotlib.pyplot as plt
import random
import queue
import heapq
import copy
import time


#from functions.baseTrain import baseTrain
#from functions.errorInFit import errorInFit
#from functions.functionPrediction import functionPrediction
#from functions.kerNameToValue import kerNameToValue
from functions.kerVal import kerVal
from functions.pool import pool

class KSearch:

    def __init__(self,xData,yData,trainSetRatio,validationSetRatio,poolSize,replica,bestClassSize,goalDepth):
        self.poolSize=poolSize
        self.replica=replica
        self.trainSetLen=int(trainSetRatio*len(xData))
        self.validationSetLen=int(validationSetRatio*(len(xData)-self.trainSetLen))
        self.xTrain=xData[:self.trainSetLen]
        self.yTrain=yData[:self.trainSetLen]
        self.xValid=xData[self.trainSetLen:self.trainSetLen+self.validationSetLen]
        self.yValid=yData[self.trainSetLen:self.trainSetLen+self.validationSetLen]
        self.xTest=xData[self.trainSetLen+self.validationSetLen:]
        self.yTest=yData[self.trainSetLen+self.validationSetLen:]
        self.xData=xData
        self.yData=yData
        self.kernelPool=pool(poolSize,self.xTrain,self.yTrain)
        self.localRatio = (trainSetRatio/(validationSetRatio*(1-trainSetRatio)))
        self.trainSetTrust= self.localRatio/(1+self.localRatio)
        self.dataSD=np.sqrt(np.var(self.yTrain))
        self.bestClassSize=bestClassSize
        self.goalDepth=goalDepth
        
    def treeSearch(self):
        root =[]
        errors=[]
        appendings=[]
        mainAppendings=[]
        fringe=[]
        plotNode=[]
        heapq.heappush(fringe,(100,[root]))
        loop=0
        aCost=0
        errorMin=100
        xContinuous=np.linspace(self.xData[0],self.xData[-1],500)
        
        
        while fringe:
            loop+=1
            #print
            #print 'Number of The Nodes Explored=',loop
            current=heapq.heappop(fringe)
            error=current[0]
            currentNode=current[1]
            label=currentNode[0]
            depth=len(label)
            #print 'depth=%i/%i'%(depth,self.goalDepth)
            #print '(label,cost,coef)=',currentNode
            
            
            if error<errorMin:
                bestNode=current
                errorMin=error
            
            if (loop==10):
                label=bestNode[1][0]
                coefs=bestNode[1][2]
                error=bestNode[0]
                predict=self.prediction(label,coefs,xContinuous)                  
                #print   '\n' ,'Best Kernel Set Found',bestNode,'\n','Functions=',self.labelToFunction(label)
                #self.plot(xContinuous,predict)
                return (error,predict)
                
            if loop>1:
                aCostNode=heapq.heappop(fringe)
                aCost=3*(aCostNode[0]-error)
                #print 'aCost',aCost
                heapq.heappush(fringe,(aCostNode[0],aCostNode[1]))
                       
            successors=self.getSuccessors(label,appendings,aCost)
            for newNode in successors:
                error=newNode[1]
                if not np.isnan(error):
                    heapq.heappush(fringe,(error,newNode))
                    
            if loop==1:
                appendings=[fringe[i][1][0] for i in range(int(len(fringe)/50.))]   
                #print 'appendingsLen=',len(appendings)
        
            '''
            appendings=[mainAppendings[k] for k in range(int(len(mainAppendings)*((depth)*1./(self.goalDepth+1))),int(len(mainAppendings)*((depth+1)*1./(self.goalDepth+1))))]
            '''
    def getSuccessors(self,nodeLabel,appendings,aCost):
        #print 'appendLen=',len(appendings)
        successors=[]
                 
        if len(nodeLabel)==0:
            for ker in range(self.poolSize):
                coefs=self.training([ker],self.xTrain,self.yTrain)
                heuristic=self.heuristic([ker],coefs)
                labelCostCoef=([ker],heuristic+aCost,coefs)
                successors.append(labelCostCoef)     
        else:
            for ker in appendings:
                label=copy.deepcopy(nodeLabel)                
                label.append(ker[0])
                coefs=self.training(label,self.xTrain,self.yTrain)                
                depth=len(label)
                heuristic=self.heuristic(label,coefs)
                labelCostCoef=(label,heuristic+aCost,coefs)
                successors.append(labelCostCoef)
        return successors

        
    def heuristic(self,label,coefs):
        heuristic=(self.trainSetTrust)*self.jTrain(label,coefs)+(1-self.trainSetTrust)*self.jValid(label,coefs)
        return heuristic
        
        
    def jTrain(self,label,coefs):
        trainPredict=self.prediction(label,coefs,self.xTrain)
        return np.sqrt(np.var(trainPredict-self.yTrain))


    def jValid(self,label,coefs):
        validPredict=self.prediction(label,coefs,self.xValid)
        return np.sqrt(np.var(validPredict-self.yValid))
        
    def training(self,label,xTrain,yTrain):
        kernels=self.labelToFunction(label)
        kerMat=[[kerVal(x,kernel) for kernel in kernels] for x in xTrain]
        kerMatInv=np.linalg.pinv(kerMat)
        coefs=np.dot(kerMatInv,yTrain)    
        return coefs

    def prediction(self,label,coefs,xData):
        kernels=self.labelToFunction(label)
        return [sum([coefs[j]*kerVal(xData[i],kernels[j]) for j in range(len(kernels))]) for i in range(len(xData))]

    def labelToFunction(self,label):
        return ['one']+[self.kernelPool[i] for i in label]

    def findBaseFunction(self):
        return self.treeSearch()

    def plot(self,xContinuous,predict,clcTime):
         '''
         fig=plt.figure()
         ax = fig.add_subplot(111)
         fig.plot(xContinuous, predict, 'b-')
         return fig
         '''
         fig=plt.figure()
         for i in range(len(predict)):
             plt.plot(xContinuous,predict[i])

         plt.plot(self.xTrain,self.yTrain,'r.')
         plt.plot(self.xValid,self.yValid,'g.')
         plt.plot(self.xTest,self.yTest,'k.')
         plt.legend(['run1','run2','run3','run4','run5','training points','validation points','test points'],prop={'size': 6})
         plt.title('poolSize=%i,replica=%i,clcTimePerRun=%.2f s'%(self.poolSize,self.replica,clcTime))
         #plt.savefig('C:\Users\AsusIran\Documents\Documents\work\projects\phd_projects\old_projects\KSearch\project/result/data{sin*exp}_poolSize=%i_replica=%i_clcTimePerRun=%i'%(self.poolSize,self.replica,clcTime) )
         return fig









        

