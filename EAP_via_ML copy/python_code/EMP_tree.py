"""
This version: August 3, 2021

Purpose: Replication of the "Empirical Asset Pricing via Machine Learning" (2018)
For all tree models and neural net models

@author: Xia Zou
"""

####### Importing package
###If you don't have these packages, please install them first
import string
import os

import pandas as pd
import datetime
import numpy as np
import statsmodels.api as sm
import math
from scipy.linalg import svd
import Functions as Fun
from sklearn import ensemble
import scipy.linalg as la
import scipy.sparse.linalg as sla
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from sklearn import metrics
import mlsauce as ms
import torch
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



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

title = dirdata + '/tree'
###add later : if not exists, then excute
###Create a folder for storing the results of r-square of different methods. if exists, comment out this code
os.mkdir(title)


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



######Star to train  ####################


################For tree model##########################
###oos r2
r2_oos=np.zeros((3,1))
###is r2
r2_is=np.zeros((3,1))

###Number of predictors
nump = xtrain.shape[1]
######num of predictors to sample
lamv = np.arange(10,nump,50)
###try
#lamv = 100

 ##num of trees
ne=100
####maximum of split
lamc = [2,4,8,16,32]
####try
#lamc = 8;
r=np.zeros((len(lamv),len(lamc),3))

for n1 in range(len(lamv)):
    nf=lamv[n1]
    for n2 in range(len(lamc)):
        nn = lamc[n2]
        clf = ensemble.RandomForestRegressor(n_estimators = ne,max_features = nf,max_depth= nn)

        clffit = clf.fit(xtrain,np.ravel(ytrain))
        yhatbig1 = clffit.predict(xtest)
        r[n1,n2,0]=1 - np.linalg.norm(yhatbig1-ytest['RET'])**2 / np.linalg.norm((ytest['RET']-mtrain.values))**2
        yhatbig1= clffit.predict(xoos)
        r[n1,n2,1]=1-np.linalg.norm((yhatbig1-yoos['RET']))**2/np.linalg.norm((yoos['RET']-mtrain.values))**2
        yhatbig1= clffit.predict(xtrain)
        r[n1,n2,2]=1-np.linalg.norm((yhatbig1-ytrain['RET']))**2/np.linalg.norm((ytrain['RET']-mtrain.values))**2


fw_2 = Fun.fw2(r[:,:,0])

r2_oos[0]=r[fw_2[0],fw_2[1],1]
r2_is[0]=r[fw_2[0],fw_2[1],2]
print('RF R2 : '+np.str0(r2_oos[0]) )





#### GBRT

lamv=np.arange(-1,0,0.2)
#lamv = -0.6

r=np.zeros((len(lamv),3))

for n1 in range(len(lamv)):
    lr=10**lamv[n1]
    alpha=2
    ne=50
    clf = ensemble.GradientBoostingRegressor(loss = 'ls',learning_rate =lr,n_estimators = ne,criterion = 'friedman_mse',max_depth = 2)
    clffit = clf.fit(xtrain,np.ravel(ytrain))
    yhatbig1 = clffit.predict(xtest)
    r[n1,0]=1 - np.linalg.norm(yhatbig1-ytest['RET'])**2 / np.linalg.norm((ytest['RET']-mtrain.values))**2
    yhatbig1= clffit.predict(xoos)
    r[n1,1]=1-np.linalg.norm((yhatbig1-yoos['RET']))**2/np.linalg.norm((yoos['RET']-mtrain.values))**2
    yhatbig1= clffit.predict(xtrain)
    r[n1,2]=1-np.linalg.norm((yhatbig1-ytrain['RET']))**2/np.linalg.norm((ytrain['RET']-mtrain.values))**2




r2_oos[1]=r[np.argmax(r[:,0]),1]
r2_is[1]=r[np.argmax(r[:,0]),2]
print('GBRT R2 : '+np.str0(r2_oos[1]) )







pathr=title+'/roos'
pathb=pathr+'.csv'
r2_oos = pd.DataFrame(r2_oos)
r2_oos.to_csv(pathb)

pathr=title+'/ris'
pathb=pathr+'.csv'
r2_is = pd.DataFrame(r2_is)
r2_is.to_csv(pathb)





########################For Neural net model ########################

###NN1: [32]
##NN2: [32,16]
##NN3: [32,16,8]
##NN4: [32,16,8,4]
##NN5: [32,16,8,4,2]
##activation function: ReLU



###try
from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda, Compose




device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))
