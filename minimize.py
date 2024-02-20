class MinimizeProblem:

    def __init__(self,xData,yData,numKer,numRep,trainSetRatio,mode='decay'):
        self.xData=xData
        self.yData=yData
        self.numKer=numKer
        self.numRep=numRep
        self.trainSetLen=int(trainSetRatio*len(xData))  
        self.mode=mode
        
    def hypothesis(self,x,parms,mode):
        if mode=='decay':
            c=parms[0:self.numKer]
            alpha=parms[self.numKer:2*self.numKer]
            xi=parms[2*self.numKer:3*self.numKer]
            return sum( [c[i]* x**(-alpha[i]) * np.exp(-x/xi[i]) for i in range(self.numKer)] )
        elif mode=='decayGrowth':
            c=parms[0:self.numKer]
            alpha=parms[self.numKer:2*self.numKer]
            beta=parms[2*self.numKer:3*self.numKer]
            gamma=parms[3*self.numKer:4*self.numKer]
            return sum( [c[i]* x**(alpha[i])* np.log(beta[i]+x)* np.tanh(gamma[i]*x)  for i in range(self.numKer)] )

    def cost(self,parms):
        return sum([(self.hypothesis(self.xData[i],parms,self.mode)-self.yData[i])**2 for i in range(self.trainSetLen)])

    def parameters(self):
        if self.mode=='decay':
            initParms=np.random.rand(3*self.numKer)
        elif self.mode=='decayingGrowth':
            initParms=np.random.rand(4*self.numKer)    
        option={'maxiter': 10000 ,'eps':1e-5,'disp':False}
        result=scipy.optimize.minimize(self.cost,initParms, method='L-BFGS-B',jac=None,options=option)
        return result.x
        
    def prediction(self):           
        parms=[(self.parameters()) for i in range(self.numRep)]
        J=[[self.hypothesis(x,parms[i],self.mode) for x in self.xData] for i in range(self.numRep)]
        return np.mean(J,axis=0)

    
        
    def plot(self,data):
        if data==None:
            plt.plot(self.xData,self.yData,'+')
            plt.plot(self.xData,self.prediction())
        else:
            plt.plot(self.xData,data)
        plt.show()
        
        
        
        
