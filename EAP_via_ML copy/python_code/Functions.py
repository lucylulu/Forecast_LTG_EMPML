
###################################
#This version: July 26, 2021
#Author: Xia Zou
#Replication of the "Empirical Asset Pricing via Machine Learning" (2018)
#Define relevant functions
###################################

####Import necessary package
import numpy as np
import math
import pandas as pd
###sq.m
#%%
def sq(a,b,step):
    r = a
    new = a
    for i in range(10000):
        new = new + step
        if new <= b:
            r = r+ new
        else:
            break
    return r
##print(sq(1,100,10) )
#%%


###soft_threshodr

def soft_threshodr(groups,nc,w,mu):
    return w/(1+mu)



###soft_threshodl

def soft_threshodl(groups,nc,w,mu):
    val =[float(np.sign(w1)*np.max(np.abs(w1)-mu,0)) for w1 in w]
    return np.array(val).reshape((len(val),1))

#### func: soft_threshode

def soft_threshode(groups,nc,w,mu):
    val =[float(np.sign(w1)*np.max(np.abs(w1)-0.5*mu,0))/(1+0.5*mu) for w1 in w]
    return np.array(val).reshape((len(val),1))
###soft_threshodg

def soft_threshodg(groups,nc,w,mu):
    w1 = w
    for i in range(nc):
        ind = (groups == i)
        wg = w1[ind,:]
        nn= np.size(wg)
        n2 = math.sqrt(sum(pow(wg,2)))
        if n2 < mu :
            w1[ind,:] = zeros(nn,1)
        else:
            w1[ind,:] = wg - mu*wg/n2

    return w1


####lossh
def lossh(y,yhat,mu):
    r= abs(yhat['RET']-y)
    l= np.zeros(len(r))
    ind = (r>mu)
    l[ind] = 2*mu*r[ind]-mu*mu
    ind = (r<= mu)
    l[ind]  = r[ind]*r[ind]
    return np.mean(l)



####f_grad
def f_grad(XX,XY,w):
    return (np.matmul(XX,w) - XY)[0]

####f_gradh
def f_gradh(w,X,y,mu):
    r = np.matmul(X,w)-y['RET']
    ind0 = np.where(abs(r)<= mu)
    ind0 = pd.Series(np.asarray(ind0)[0])
    ind1 = np.where(r>mu)
    ind1 = pd.Series(np.asarray(ind1)[0])

    indf1 = np.where(r< (-mu))
    indf1 = pd.Series(np.array(indf1)[0])
    grad = np.matmul(np.transpose(X.loc[ind0,:]),(np.matmul(X.loc[ind0,:],w)-y.loc[ind0,'RET']))+np.matmul(np.transpose(mu*X.loc[ind1,:]),np.ones(len(ind1)))-np.matmul(np.transpose(mu*X.loc[indf1,:]),np.ones(len(indf1)))
    return grad

### proximal_l   only for lasso

def proximalH_l(groups,nc,xtest,mtrain,ytest,w,X,y,mu,tol,L,l2,func):
    w= np.asarray(w[0])
    dim = X.shape[0]
    max_iter = 3000
    gamma= 1/L
    l1 = l2

    v= w
    yhatbig1=np.matmul(xtest,w) + mtrain.values
    r20=lossh(yhatbig1,ytest,mu)
    for t in range(max_iter):
        vold=v
        w_perv=w
        w= np.asarray(v-(gamma*f_gradh(v,X,y,mu)))
        mu1=l1*gamma
        w=func(groups,nc,w,mu1)
        w = np.asarray([float(w1) for w1 in w])
        v=w + t/(t+3)*(w-w_perv)
        if ((np.linalg.norm(v-vold)**2) < ((np.linalg.norm(vold)**2)*tol) or np.sum(abs(v-vold))==0):
            break
    return v


###proximalH

def proximalH(groups,nc,xtest,mtrain,ytest,w,X,y,mu,tol,L,l2,func):
    dim = X.shape[0]
    max_iter = 3000
    gamma= 1/L
    l1 = l2
    v= w
    yhatbig1=np.matmul(xtest,w) + mtrain.values
    r20=lossh(yhatbig1,ytest,mu)
    for t in range(max_iter):
        vold=v
        w_perv=w
        w=v-gamma*f_gradh(v,X,y,mu)
        mu1=l1*gamma
        w=func(groups,nc,w,mu1)
        v=w+t/(t+3)*(w-w_perv)
        if (sum(pow(v-vold,2)) < (sum(pow(vold,2))*tol) or sum(abs(v-vold))==0):
            break
    return v





###Func pls

def pls(X,y,A):
    s = np.matmul(np.transpose(X),y)[0]
    R = np.zeros((X.shape[1],A))
    TT = np.zeros((X.shape[0],A))
    P = np.zeros((X.shape[1],A))
    U = np.zeros((X.shape[0],A))
    V = np.zeros((X.shape[1],A))
    B = np.zeros((X.shape[1],A))
    Q = np.zeros((1,A))

    for i in range(A):
        i= i-1
        q= np.matmul(np.transpose(s),s)
        r = q*s
        t=np.matmul(X,r)
        t=t-np.mean(t);
        normt= np.linalg.norm(t)
        t=t/normt
        r=r/normt
        p=np.matmul(np.transpose(X),t)
        q=np.matmul(np.transpose(y),t)
        u=q[0]*y
        v=p
        if i> (-1) :
            v= v-np.matmul(V[:,range(i)],(np.matmul(np.transpose(V[:,range(i)]),p)))
            u= u-np.matmul(TT[:,range(i)],(np.matmul(np.transpose(TT[:,range(i)]),u)))

        v= v/np.linalg.norm(v)
        s=s-v*np.matmul(np.transpose(v),s)

        R[:,i+1] = r
        TT[:,i+1] =t
        P[:,i+1]=p
        U[:,i+1] = u['RET']
        V[:,i+1] =v
        Q[:,i+1]=q

    for i in range(A-1):
        C = np.matmul(R[:,range(i+1)],np.transpose(Q[:,range(i+1)]))
        B[:,i+1] = C[:,0]

    return B


#####Def proximal

def proximal(groups,nc,XX,XY,tol,L,l2,func):
   dim = XX.shape[0]
   max_iter = 3000
   gamma= 1/L
   l1 = l2

   w = np.zeros((dim,1))
   v = w

   for t in range(max_iter):
       vold=v
       w_perv=w
       w=np.transpose(v.reshape(1,v.size)-(gamma*f_grad(XX,XY,v)).values)
       w=func(groups,nc,w,l1*gamma)
       v=w + t/(t+3)*(w-w_perv)
       if ((np.linalg.norm(v-vold)**2 < (np.linalg.norm(vold)**2)*tol) or (np.sum(abs(v-vold))==0)):
           break

   return v

###Fw2

def fw2(x):
    x= r[:,:,0]
    maximum = max(map(max,x))
    p = np.where(x == maximum)
    return(p)


if __name__ =='__main__':
    soft_threshodr()
    soft_threshode()
    soft_threshodl()
    soft_threshodg()
    lossh()
    f_grad()
    f_gradh()
    proximalH_l()
    proximalH()
    proximal_l()
    proximal()
    pls()
    fw2()






# ww = np.array([[1,2,3],[4,5,6]])
#
# groups = 0
# nc= 1
# w= ww
# mu=1
# i=0
#soft_threshodg(0,1,ww,1)
# print(np.zeros(len(ww[1])))
#
#
# np.size(ww[1,:])
#
# print(soft_threshodl(0.2,100,2,1))
# sum(ww[1,:])
# math.sqrt(sum(pow(ww[1,:],2)))
