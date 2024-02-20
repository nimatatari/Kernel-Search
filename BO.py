import numpy as np

def f(x):
    return np.sin(3*x)+np.cosh(x)


    
XO=[-1,0,1]
YO=[f(x) for x in x0]

Xo=np.linspace(-5,5,20)

def ker(x,z):
    h=1
    w=1
    r=x-z
    return h**2 *np.exp(-.5*(r/w)**2)

def mu(x):
    return 0


muO=[mu(x) for x in XO]
muo=[mu(x) for x in Xo]

KOO=[[ker(x,z) for z in XO] for x in XO]
kOOinv=np.linalg.inv(KOO)
KOo=[[ker(x,z) for z in Xo] for x in XO]
KoO=[[ker(x,z) for z in XO] for x in Xo]
Koo=[[ker(x,z) for z in Xo] for x in Xo]


mo=muo+np.dot(np.dot(KoO,KOOinv),(YO-muO))
Coo=Koo-np.dot(KoO,KOOinv)
