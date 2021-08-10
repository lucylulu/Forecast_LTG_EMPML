"""
This version: July 27, 2021

Purpose: Replication of the "Empirical Asset Pricing via Machine Learning" (2018)
For all regression model

@author: Xia Zou
"""
import string
import os

import pandas as pd
import datetime
import numpy as np
import statsmodels.api as sm
import math
from scipy.linalg import svd
import Functions as Fun

import scipy.linalg as la
import scipy.sparse.linalg as sla



####Set the path to where your codes located
os.chdir('/Users/xiazou/Desktop/Tinbergen_Study/block5/Forecasting_LTG/Replication/EAP_via_ML/python_code')
###path of the function file

###Path where the data locate
path='/Users/xiazou/Desktop/Tinbergen_Study/block5/Forecasting_LTG/Replication/EAP_via_ML'
# set your own folder path (the same with R folder path)
dirdata=path+'/Data_Cleaned'

###Some parameters
mu=0.2;
tol=1e-10;


###Make a dir for storing results

title = dirdata + '/Reg'
###add later : if not exists, then excute
###Create a folder for storing the results of r-square of different methods. if exists, comment out this code
os.mkdir(title)
titleB =title+'/B'
###Create a folder for storing the results of coefficients of different methods. if exists, comment out this code
os.mkdir(titleB);


#####Read files
pathY = dirdata + '/Y_cleaned.csv'
pathX = dirdata + '/X_cleaned.csv'
pathdate = dirdata + '/Date_list.csv'

Y_all = pd.read_csv(pathY)

X_all = pd.read_csv(pathX)

Date_index = pd.read_csv(pathdate)

Date_index['DATE.x']=pd.to_datetime(Date_index['DATE.x'])

#######Training, validation, testing set

######Start date = 1985-01 , end date =

####for training

start_date = pd.to_datetime('1957-03-01')

end_date =  start_date + pd.offsets.DateOffset(years=18)

ind_training = (start_date < Date_index['DATE.x']) & (Date_index['DATE.x'] < end_date)

ytrain = Y_all[ind_training]
xtrain = X_all[ind_training]

####for validation

start_date =  end_date
end_date = start_date + pd.offsets.DateOffset(years=12)

ind_vali = (start_date < Date_index['DATE.x']) & (Date_index['DATE.x'] < end_date)

ytest = Y_all[ind_vali]
xtest= X_all[ind_vali]

#####for test  oos

start_date =  end_date
end_date = pd.to_datetime('2016-12-30')

ind_test = (start_date < Date_index['DATE.x']) & (Date_index['DATE.x'] < end_date)
#print(type(ind_test))
yoos = Y_all[ind_test]
xoos= X_all[ind_test]

#######Monthly Demean %%%
ytrain_demean= ytrain - np.mean(ytrain);
ytest_demean=ytest-np.mean(ytest);
mtrain=np.mean(ytrain);
mtest=np.mean(ytest);

sd=np.zeros(len(xtrain.columns)) #dim of sd? sd for each characteristics
for i in range(len(xtrain.columns)):
    s=np.std(xtrain.iloc[:,[i]])
    colnames = xtrain.columns
    if s.values > 0:
        colname = colnames[i]
        xtrain.loc[:,colname]=xtrain[colname]/s.values
        xtest.loc[:,colname]=xtest.loc[:,colname]/s.values
        xoos.loc[:,colname]=xoos.loc[:,colname]/s.values
        sd[i]=s.values


XX =np.matmul(np.transpose(xtrain),xtrain)
U,S,V=svd(XX)
#### singular value decomposition such that XX=U*S*V'
L=S[0]
###disp 'Lasso L = '
####disp(L)
Y=ytrain_demean;
XY= np.matmul(np.transpose(xtrain),Y)

 #### Start to Train %%%

 ##### OLS %%%
 r2_oos=np.zeros(12) #### OOS R2
 r2_is=np.zeros(12) #####IS R2
 modeln=0   #####Basic OLS model
 groups=0
 nc=0
  #model fitting

 clf=sm.OLS(ytrain_demean,xtrain)
 yhatres = clf.fit()
 yhatbig1= yhatres.predict(xoos)+ mtrain.values
 #prediction for oos
 r2_oos[modeln]=1-sum((yhatbig1-yoos['RET'])**2)/sum((yoos['RET']-mtrain.values)**2) # oos r2

 yhatbig1=yhatres.predict(xtrain) + mtrain.values
 r2_is[modeln]=1-(yhatbig1-ytrain['RET']).pow(2).sum()/sum(pow(ytrain['RET']-mtrain.values,2))
 b= pd.DataFrame(b)
 pathb=title + '/B/b'
 pathb= pathb + np.str0(modeln)
 pathb= pathb + '.csv'
 b.to_csv(pathb)
 print('Simple OLS R2 : '+np.str0(r2_oos[modeln]) )


#model 2 Simple OLS + H
 modeln=modeln+1

 b=Fun.proximalH_l(groups,nc,xtest,mtrain,ytest,b,xtrain,ytrain_demean,mu,tol,L,0,Fun.soft_threshodl)
 b= pd.DataFrame(b)
 yhatbig1=np.matmul(xoos,b)+mtrain.values

 r2_oos[modeln]=1- np.linalg.norm(yhatbig1[0]-yoos['RET'])**2 / np.linalg.norm((yoos['RET']-mtrain.values))**2
 yhatbig1 = np.matmul(xtrain,b) + mtrain.values
 r2_is[modeln] = 1 - np.linalg.norm(yhatbig1[0]-ytrain['RET'])**2 / np.linalg.norm((ytrain['RET']-mtrain.values))**2
 pathb=title + '/B/b'
 pathb= pathb + np.str0(modeln)
 pathb= pathb + '.csv'
 b.to_csv(pathb)
 print('Simple OLS+H R2 : '+np.str0(r2_oos[modeln]) )















#model 3 PCR
 modeln=modeln+1
 ne=30;
 #XX=xtrain.'*xtrain;
 pca_val,pca_vec=la.eig(XX)
 p1=pca_vec[:,list(range(ne))]
 Z=np.matmul(xtrain,p1)

 r=np.zeros((3,ne))
 B=np.zeros((xtrain.shape[1],ne))
 Y=ytrain_demean

 for j in range(ne-1):
     j=j+1
     xx=Z[range(j)]
     b = np.matmul(np.matmul(np.linalg.inv(np.matmul(np.transpose(xx),xx)),np.transpose(xx)),Y)
     b=np.matmul(p1[:,range(j)],b)
     yhatbig1=np.matmul(xtest,b)+mtrain.values
    # r[0,j-1] = 1 - (np.sum((yhatbig1[0]-ytest['RET'])**2)/np.sum((ytest['RET']-mtrain.values**2)))
     r[0,j-1]=1-(np.linalg.norm((yhatbig1[0]-ytest['RET'])))**2/(np.linalg.norm((ytest['RET']-mtrain.values)))**2
     yhatbig1=np.matmul(xoos,b)+mtrain.values
     r[1,j-1]=1-(np.linalg.norm((yhatbig1[0]-yoos['RET'])))**2/(np.linalg.norm((yoos['RET']-mtrain.values)))**2
     yhatbig1=np.matmul(xtrain,b)+mtrain.values
     r[2,j-1]=1-(np.linalg.norm((yhatbig1[0]-ytrain['RET'])))**2/(np.linalg.norm((ytrain['RET']-mtrain.values)))**2
     B[:,j-1]= b[0]




 b=np.zeros((xtest.shape[1],1))
 j=ne-1;
 yhatbig1=np.matmul(xtest,b)+mtrain.values
 r[0,j]=1-(np.linalg.norm(yhatbig1[0]-ytest['RET']))**2/np.linalg.norm((ytest['RET']-mtrain.values))**2
 yhatbig1=np.matmul(xoos,b)+mtrain.values
 r[1,j]=1-np.linalg.norm((yhatbig1[0]-yoos['RET']))**2/np.linalg.norm((yoos['RET']-mtrain.values))**2
 yhatbig1=np.matmul(xtrain,b)+mtrain.values
 r[2,j]=1-np.linalg.norm((yhatbig1[0]-ytrain['RET']))**2/np.linalg.norm((ytrain['RET']-mtrain.values))**2
 B[:,j]=b[0]

 r2_oos[modeln]=r[1,np.argmax(r[0,:])]
 r2_is[modeln]=r[2,np.argmax(r[0,:])]
 b=pd.DataFrame(B[:,np.argmax(r[0,:])] )
 pathb=title + '/B/b'
 pathb= pathb + np.str0(modeln)
 pathb= pathb + '.csv'
 b.to_csv(pathb)
 print('PCR R2 : '+np.str0(r2_oos[modeln]) )

###%Model 4 PLS
modeln=modeln+1
B=Fun.pls(xtrain,ytrain_demean,30)
ne=30
r=np.zeros((3,ne))
Y=ytrain_demean

 for j in range(ne):
     j=0
     b=B[:,j]
     yhatbig1 = np.matmul(xtest,b)+mtrain.values
     r[0,j]=1 - np.linalg.norm(yhatbig1-ytest['RET'])**2 / np.linalg.norm((ytest['RET']-mtrain.values))**2
     yhatbig1=np.matmul(xoos,b)+mtrain.values
     r[1,j]=1-np.linalg.norm((yhatbig1-yoos['RET']))**2/np.linalg.norm((yoos['RET']-mtrain.values))**2
     yhatbig1=np.matmul(xtrain,b)+mtrain.values
     r[2,j]=1-np.linalg.norm((yhatbig1-ytrain['RET']))**2/np.linalg.norm((ytrain['RET']-mtrain.values))**2

r2_oos[modeln]=r[1,np.argmax(r[0,:])]
r2_is[modeln]=r[2,np.argmax(r[0,:])]
b=pd.DataFrame(B[:,np.argmax(r[0,:])] )
pathb=title + '/B/b'
pathb= pathb + np.str0(modeln)
pathb= pathb + '.csv'
b.to_csv(pathb)
print('PLS R2 : '+np.str0(r2_oos[modeln]) )





 ##### Lasso %%%
 modeln=modeln+1   ###Model 5 LASSO
 lamv=np.arange(-2,4,0.1)
 alpha=1
 r=np.zeros((3,len(lamv)))

 for j in range(len(lamv)):
     l2=10**lamv[j]
     b=Fun.proximal(groups,nc,XX,XY,tol,L,l2,Fun.soft_threshodl);
     yhatbig1 = np.matmul(xtest,b)+mtrain.values
     r[0,j]=1 - np.linalg.norm(yhatbig1[0]-ytest['RET'])**2 / np.linalg.norm((ytest['RET']-mtrain.values))**2
     yhatbig1=np.matmul(xoos,b)+mtrain.values
     r[1,j]=1-np.linalg.norm((yhatbig1[0]-yoos['RET']))**2/np.linalg.norm((yoos['RET']-mtrain.values))**2
     yhatbig1=np.matmul(xtrain,b)+mtrain.values
     r[2,j]=1-np.linalg.norm((yhatbig1[0]-ytrain['RET']))**2/np.linalg.norm((ytrain['RET']-mtrain.values))**2



 r2_oos[modeln]=r[1,np.argmax(r[0,:])]
 r2_is[modeln]=r[2,np.argmax(r[0,:])]
 l2=10**lamv[np.argmax(r[0,:])]

 b=Fun.proximal(groups,nc,XX,XY,tol,L,l2,Fun.soft_threshodl)
 b=pd.DataFrame(b)
 pathb=title + '/B/b'
 pathb= pathb + np.str0(modeln)
 pathb= pathb + '.csv'
 b.to_csv(pathb)
 print('LASSO R2 : '+np.str0(r2_oos[modeln]) )



 ###Model 6 LASSO+H
 modeln=modeln+1
 b=Fun.proximalH_l(groups,nc,xtest,mtrain,ytest,b,xtrain,ytrain_demean,mu,tol,L,l2,Fun.soft_threshodl)
 yhatbig1 = np.matmul(xoos,b)+mtrain.values
 r2_oos[modeln]=1- np.linalg.norm(yhatbig1-yoos['RET'])**2 / np.linalg.norm((yoos['RET']-mtrain.values))**2
 yhatbig1 = np.matmul(xtrain,b) + mtrain.values
 r2_is[modeln] = 1 - np.linalg.norm(yhatbig1-ytrain['RET'])**2 / np.linalg.norm((ytrain['RET']-mtrain.values))**2
 b=pd.DataFrame(b)
 pathb=title + '/B/b'
 pathb= pathb + np.str0(modeln)
 pathb= pathb + '.csv'
 b.to_csv(pathb)
 print('LASSO+H R2 : '+np.str0(r2_oos[modeln]) )






 ####Ridge: model 7
 modeln=modeln+1
 lamv= np.arange(0,6,0.1)
 alpha=1
 r=np.zeros((3,len(lamv)))

 for j in range(len(lamv)):
     l2=10**lamv[j]
     b=Fun.proximal(groups,nc,XX,XY,tol,L,l2,Fun.soft_threshodr)
     yhatbig1 = np.matmul(xtest,b)+mtrain.values
     r[0,j]=1 - np.linalg.norm(yhatbig1[0]-ytest['RET'])**2 / np.linalg.norm((ytest['RET']-mtrain.values))**2
     yhatbig1=np.matmul(xoos,b)+mtrain.values
     r[1,j]=1-np.linalg.norm((yhatbig1[0]-yoos['RET']))**2/np.linalg.norm((yoos['RET']-mtrain.values))**2
     yhatbig1=np.matmul(xtrain,b)+mtrain.values
     r[2,j]=1-np.linalg.norm((yhatbig1[0]-ytrain['RET']))**2/np.linalg.norm((ytrain['RET']-mtrain.values))**2


 r2_oos[modeln]=r[1,np.argmax(r[0,:])]
 r2_is[modeln]=r[2,np.argmax(r[0,:])]
 l2=10**lamv[np.argmax(r[0,:])]

 b=Fun.proximal(groups,nc,XX,XY,tol,L,l2,Fun.soft_threshodr)
 b=pd.DataFrame(b)
 pathb=title + '/B/b'
 pathb= pathb + np.str0(modeln)
 pathb= pathb + '.csv'
 b.to_csv(pathb)
 print('Ridge R2 : '+np.str0(r2_oos[modeln]) )


 ####Model 8 Ridge +H
 modeln=modeln+1
 b=Fun.proximalH_l(groups,nc,xtest,mtrain,ytest,b,xtrain,ytrain_demean,mu,tol,L,l2,Fun.soft_threshodr)
 yhatbig1 = np.matmul(xoos,b)+mtrain.values
 r2_oos[modeln]=1- np.linalg.norm(yhatbig1-yoos['RET'])**2 / np.linalg.norm((yoos['RET']-mtrain.values))**2
 yhatbig1 = np.matmul(xtrain,b) + mtrain.values
 r2_is[modeln] = 1 - np.linalg.norm(yhatbig1-ytrain['RET'])**2 / np.linalg.norm((ytrain['RET']-mtrain.values))**2
 b=pd.DataFrame(b)
 pathb=title + '/B/b'
 pathb= pathb + np.str0(modeln)
 pathb= pathb + '.csv'
 b.to_csv(pathb)
 print('Ridge+H R2 : '+np.str0(r2_oos[modeln]) )


 ####Model 9 Elastic Net %%%
 modeln=modeln+1
 lamv = np.arange(-2,4,0.1)

 alpha=0.5;
 r=np.zeros((3,len(lamv)))

 for j in range(len(lamv)):
     l2=10**lamv[j]
     b=Fun.proximal(groups,nc,XX,XY,tol,L,l2,Fun.soft_threshode)
     yhatbig1 = np.matmul(xtest,b)+mtrain.values
     r[0,j]=1 - np.linalg.norm(yhatbig1[0]-ytest['RET'])**2 / np.linalg.norm((ytest['RET']-mtrain.values))**2
     yhatbig1=np.matmul(xoos,b)+mtrain.values
     r[1,j]=1-np.linalg.norm((yhatbig1[0]-yoos['RET']))**2/np.linalg.norm((yoos['RET']-mtrain.values))**2
     yhatbig1=np.matmul(xtrain,b)+mtrain.values
     r[2,j]=1-np.linalg.norm((yhatbig1[0]-ytrain['RET']))**2/np.linalg.norm((ytrain['RET']-mtrain.values))**2

 r2_oos[modeln]=r[1,np.argmax(r[0,:])]
 r2_is[modeln]=r[2,np.argmax(r[0,:])]
 l2=10**lamv[np.argmax(r[0,:])]

 b=Fun.proximal(groups,nc,XX,XY,tol,L,l2,Fun.soft_threshode)
 b=pd.DataFrame(b)
 pathb=title + '/B/b'
 pathb= pathb + np.str0(modeln)
 pathb= pathb + '.csv'
 b.to_csv(pathb)
 print('Enet R2 : '+np.str0(r2_oos[modeln]) )




 ####Model 10 Enet +H
 modeln=modeln+1
 b=Fun.proximalH_l(groups,nc,xtest,mtrain,ytest,b,xtrain,ytrain_demean,mu,tol,L,l2,Fun.soft_threshode)
 yhatbig1 = np.matmul(xoos,b)+mtrain.values
 r2_oos[modeln]=1- np.linalg.norm(yhatbig1-yoos['RET'])**2 / np.linalg.norm((yoos['RET']-mtrain.values))**2
 yhatbig1 = np.matmul(xtrain,b) + mtrain.values
 r2_is[modeln] = 1 - np.linalg.norm(yhatbig1-ytrain['RET'])**2 / np.linalg.norm((ytrain['RET']-mtrain.values))**2
 b=pd.DataFrame(b)
 pathb=title + '/B/b'
 pathb= pathb + np.str0(modeln)
 pathb= pathb + '.csv'
 b.to_csv(pathb)
 print('Enet+H R2 : '+np.str0(r2_oos[modeln]) )




print(r2_oos)
print(r2_is)
