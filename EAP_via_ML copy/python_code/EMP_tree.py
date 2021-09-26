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
os.chdir('/Users/xiazou/Desktop/Tinbergen_Study/block5/Forecasting_LTG/Replication/EAP_via_ML/python_code')
import pandas as pd
import datetime
import numpy as np
import statsmodels.api as sm
import math
from scipy.linalg import svd
import Functions as Func
from sklearn import ensemble
import scipy.linalg as la
import scipy.sparse.linalg as sla
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from time import time
from sklearn import metrics
import mlsauce as ms
import torch
from torch import nn,optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
import multiprocessing as mp
print("Number of processors: ", mp.cpu_count())

####Set the path to where your codes located
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
if not os.path.isdir(title):
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

nums= 30
start_dates = [pd.to_datetime('1957-03-01')+pd.offsets.DateOffset(years=x) for x in range(nums) ]






###Files for storing results
r2_oos_all = np.zeros((7,len(start_dates)))
r2_is_all = np.zeros((7,len(start_dates)))



for m in range(len(start_dates)):

    start_date = start_dates[m]

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
    end_date = start_date + pd.offsets.DateOffset(years=1)
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
    r2_oos=np.zeros((7,1))
    ###is r2
    r2_is=np.zeros((7,1))

    ###Number of predictors
    nump = xtrain.shape[1]
    ######num of predictors to sample
    lamv = np.arange(10,nump,50)
    ###try
    #lamv = [100]

     ##num of trees
    ne=100
    ####maximum of split

    lamc = [2,4,8,16,32]
    ####try
    #lamc = [8];
    r = np.zeros((len(lamv),len(lamc),3))

    for n1 in range(len(lamv)):
        nf=lamv[n1]
        for n2 in range(len(lamc)):
            nc = lamc[n2]
            clf = ensemble.RandomForestRegressor(n_estimators = ne,max_features = nf,max_depth= nc)

            clffit = clf.fit(xtrain,np.ravel(ytrain))
            yhatbig1 = clffit.predict(xtest)
            r[n1,n2,0]=1 - np.linalg.norm(yhatbig1-ytest['RET'])**2 / np.linalg.norm((ytest['RET']-mtrain.values))**2
            yhatbig1= clffit.predict(xoos)
            r[n1,n2,1]=1-np.linalg.norm((yhatbig1-yoos['RET']))**2/np.linalg.norm((yoos['RET']-mtrain.values))**2
            yhatbig1= clffit.predict(xtrain)
            r[n1,n2,2]=1-np.linalg.norm((yhatbig1-ytrain['RET']))**2/np.linalg.norm((ytrain['RET']-mtrain.values))**2

    fw_2 = Func.fw2(r[:,:,0])
    r2_oos[0]=r[fw_2[0][0],fw_2[1][0],1]
    r2_is[0]=r[fw_2[0][0],fw_2[1][0],2]
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













    ########################For Neural net model ########################

    ###NN1: [32]
    ##NN2: [32,16]
    ##NN3: [32,16,8]
    ##NN4: [32,16,8,4]
    ##NN5: [32,16,8,4,2]
    ##activation function: ReLU




    ###Transform the data
    xtrain= torch.tensor(np.array(xtrain))
    ytrain = torch.tensor(np.array(ytrain))
    xtest = torch.tensor(np.array(xtest))
    ytest = torch.tensor(np.array(ytest))
    xoos = torch.tensor(np.array(xoos))
    yoos = torch.tensor(np.array(yoos))

    ###Set parameters


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('Using {} device'.format(device))


    ###NUmber of features
    N_fetures = xtrain.shape[1]
    ##Set for criterion function
    criterion = nn.MSELoss()
    ##Set up for optimizer
    #optimizer = optim.Adam(model.parameters(),lr=0.001)


    xtrain = xtrain.to(device)
    ytrain = ytrain.to(device)
    xtest = xtest.to(device)
    ytest = ytest.to(device)

    criterion = criterion.to(device)


    train_ds = TensorDataset(xtrain, ytrain)
    #train_dl = DataLoader(train_ds, batch_size=bs)
    test_ds = TensorDataset(xtest, ytest)
    #test_dl = DataLoader(test_ds, batch_size=bs)
    oos_ds = TensorDataset(xoos, yoos)


    #####Start training NN1 model 3
    ##Learning
    lamv=np.arange(-3,-2,0.02)
    #lamv = -0.6

    r=np.zeros((len(lamv),3))


    for n1 in range(len(lamv)):
        lr=10**lamv[n1]
        bs = 10000
        epochs = 100
        train_dl= DataLoader(train_ds,bs)
        model = Func.NeuralNetwork1(N_fetures)
        model = model.to(device)

        optimizer =  optim.Adam(model.parameters(),lr)
        ###Train NN1 model
        for epoch in range(epochs):
            for xb,yb in train_dl:
                y_pred = model(xb.float())
                train_loss = criterion(y_pred, yb.float())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtest,ytest)
            test_loss = losses*nums

        r[n1,0]=1 - test_loss/ np.linalg.norm((ytest-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xoos,yoos)
            oos_loss = losses*nums
        r[n1,1]=1 - oos_loss/ np.linalg.norm((yoos-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtrain,ytrain)
            train_loss = np.sum(np.multiply(losses, nums))
        r[n1,2]=1 - train_loss/ np.linalg.norm((ytrain-mtrain.values))**2




    r2_oos[2]=r[np.argmax(r[:,0]),1]
    r2_is[2]=r[np.argmax(r[:,0]),2]
    print('NN1 R2 : '+np.str0(r2_oos[2]) )


    #####Start training NN2 model 4
    ##Learning
    lamv=np.arange(-3,-2,0.02)
    #lamv = -0.6

    r=np.zeros((len(lamv),3))


    for n1 in range(len(lamv)):
        lr=10**lamv[n1]
        bs = 10000
        epochs = 100
        train_dl= DataLoader(train_ds,bs)
        model = Func.NeuralNetwork2(N_fetures)
        optimizer =  optim.Adam(model.parameters(),lr)
        ###Train NN1 model
        for epoch in range(epochs):
            for xb,yb in train_dl:
                y_pred = model(xb.float())
                train_loss = criterion(y_pred, yb.float())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtest,ytest)
            test_loss = losses*nums
        r[n1,0]=1 - test_loss/ np.linalg.norm((ytest-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xoos,yoos)
            oos_loss = losses*nums
        r[n1,1]=1 - oos_loss/ np.linalg.norm((yoos-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtrain,ytrain)
            train_loss = np.sum(np.multiply(losses, nums))
        r[n1,2]=1 - train_loss/ np.linalg.norm((ytrain-mtrain.values))**2



    r2_oos[3]=r[np.argmax(r[:,0]),1]
    r2_is[3]=r[np.argmax(r[:,0]),2]
    print('NN2 R2 : '+np.str0(r2_oos[3]) )



    #####Start training NN3 model 5
    ##Learning
    lamv=np.arange(-3,-2,0.02)
    #lamv = -0.6

    r=np.zeros((len(lamv),3))



    for n1 in range(len(lamv)):
        lr=10**lamv[n1]
        bs = 10000
        epochs = 100
        train_dl= DataLoader(train_ds,bs)
        model = Func.NeuralNetwork3(N_fetures)
        optimizer =  optim.Adam(model.parameters(),lr)
        ###Train NN1 model
        for epoch in range(epochs):
            for xb,yb in train_dl:
                y_pred = model(xb.float())
                train_loss = criterion(y_pred, yb.float())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtest,ytest)
            test_loss = losses*nums
        r[n1,0]=1 - test_loss/ np.linalg.norm((ytest-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xoos,yoos)
            oos_loss = losses*nums
        r[n1,1]=1 - oos_loss/ np.linalg.norm((yoos-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtrain,ytrain)
            train_loss = np.sum(np.multiply(losses, nums))
        r[n1,2]=1 - train_loss/ np.linalg.norm((ytrain-mtrain.values))**2




    r2_oos[4]=r[np.argmax(r[:,0]),1]
    r2_is[4]=r[np.argmax(r[:,0]),2]
    print('NN3 R2 : '+np.str0(r2_oos[4]) )


    #####Start training NN4 model 6
    ##Learning

    lamv=np.arange(-3,-2,0.02)
    #lamv = -0.6

    r=np.zeros((len(lamv),3))



    for n1 in range(len(lamv)):
        lr=10**lamv[n1]
        bs = 10000
        epochs = 100
        train_dl= DataLoader(train_ds,bs)
        model = Func.NeuralNetwork4(N_fetures)
        optimizer =  optim.Adam(model.parameters(),lr)
        ###Train NN1 model
        for epoch in range(epochs):
            for xb,yb in train_dl:
                y_pred = model(xb.float())
                train_loss = criterion(y_pred, yb.float())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtest,ytest)
            test_loss = losses*nums
        r[n1,0]=1 - test_loss/ np.linalg.norm((ytest-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xoos,yoos)
            oos_loss = losses*nums
        r[n1,1]=1 - oos_loss/ np.linalg.norm((yoos-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtrain,ytrain)
            train_loss = np.sum(np.multiply(losses, nums))
        r[n1,2]=1 - train_loss/ np.linalg.norm((ytrain-mtrain.values))**2




    r2_oos[5]=r[np.argmax(r[:,0]),1]
    r2_is[5]=r[np.argmax(r[:,0]),2]
    print('NN4 R2 : '+np.str0(r2_oos[5]) )



    #####Start training NN5 model 7
    ##Learning
    lamv=np.arange(-3,-2,0.02)
    #lamv = -0.6

    r=np.zeros((len(lamv),3))



    for n1 in range(len(lamv)):
        lr=10**lamv[n1]
        bs = 10000
        epochs = 100
        train_dl= DataLoader(train_ds,bs)
        model = Func.NeuralNetwork5(N_fetures)
        optimizer =  optim.Adam(model.parameters(),lr)
        ###Train NN1 model
        for epoch in range(epochs):
            for xb,yb in train_dl:
                y_pred = model(xb.float())
                train_loss = criterion(y_pred, yb.float())
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtest,ytest)
            test_loss = losses*nums
        r[n1,0]=1 - test_loss/ np.linalg.norm((ytest-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xoos,yoos)
            oos_loss = losses*nums
        r[n1,1]=1 - oos_loss/ np.linalg.norm((yoos-mtrain.values))**2

        with torch.no_grad():
            losses, nums = Func.loss_batch(model,criterion,xtrain,ytrain)
            train_loss = np.sum(np.multiply(losses, nums))
        r[n1,2]=1 - train_loss/ np.linalg.norm((ytrain-mtrain.values))**2




    r2_oos[6]=r[np.argmax(r[:,0]),1]
    r2_is[6]=r[np.argmax(r[:,0]),2]
    print('NN5 R2 : '+np.str0(r2_oos[6]) )


    pathr=title+'/roos'+ str(m)
    pathb=pathr+'.csv'
    r2_oos = pd.DataFrame(r2_oos)
    r2_oos.to_csv(pathb)

    pathr=title+'/ris'+str(m)
    pathb=pathr+'.csv'
    r2_is = pd.DataFrame(r2_is)
    r2_is.to_csv(pathb)

    r2_is_all[:,s] = r2_is[:,0]
    r2_oos_all[:,s] = r2_oos[:,0]


title_all_oos = dirdata+'/roos_all.csv'
r2_oos_all = pd.DataFrame(r2_oos_all)
r2_oos_all_mean = r2_oos_all.mean(axis=1)
r2_oos_all.to_csv(title_all_oos)

title_all_is = dirdata+'/is_all.csv'
r2_is_all = pd.DataFrame(r2_is_all)
r2_is_all_mean = r2_is_all.mean(axis=1)
r2_is_all.to_csv(title_all)
