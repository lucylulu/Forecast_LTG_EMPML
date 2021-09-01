
"""
This version: August 20, 2021

Purpose: Replication of the Earnings-part-2

@author: Xia Zou
"""  



####remove everything
rm(list = ls()) 
library(haven)
library(randomForest)
####path of data
pathdta = '/Users/xiazou/Desktop/Tinbergen_Study/block5/Forecasting_LTG/Replication/OneDrive-2021-08-19'
setwd(pathdta)
# dataset_at_pershare = read_dta('dataset_at_pershare.dta')

###Read dataset_pershare 
dataset_pershare =  read_dta('dataset_pershare.dta')



#define function for cross validation: create a new variable to the dataset that split 
#the dataset into training and validation 

####Input variables: dataset: the dataset you want to have cross validation 
####percentage: percentage of the training set 

cv = function(dataset,percentage){  
  data_id = as.data.frame(unique(dataset$gvkey)  )
  colnames(data_id) = c('gvkey')
  
  num_id = length(data_id$gvkey) 
  data_id['wookie'] = runif(num_id) 
  ####0 is the training 
  data_id['MODELING_GROUP'] = 0  
  #####1 is for validation 
  percentage = 0.9 
  data_id[data_id$wookie > percentage,'MODELING_GROUP'] = 1
  data_out = merge(dataset,data_id,by='gvkey')
  return(data_out)
}




####Rough missing value imputation??? replace all the missing values with 0

miss_with0= function(x){
  x[is.na(x)]=0
  return(x)
}

for(i in seq(ncol(dataset_pershare))){
  dataset_pershare[,i]= miss_with0(dataset_pershare[,i])
}

# dataset_price_used =  read_dta('dataset_price_used.dta')
# dataset_unscaled = read_dta('dataset_unscaled.dta')
###extracting input variables for training 
unique(dataset_at_pershare$gvkey)

input_variables = readxl::read_xlsx('Variable_Definition_B.xlsx')
input_variables = input_variables[-102,]


colnamesdataset = colnames(dataset_pershare)


inputvar_loc = which(input_variables$Acronym %in% colnames(dataset_pershare))
####inputvar for ML only setting 
inputvar = input_variables$Acronym[inputvar_loc]

input_var_location = which(colnames(dataset_pershare) %in% input_variables$Acronym )




non_inputvar = colnames(dataset_pershare)[-input_var_location]

inputvar = c(inputvar,'ipo',"epspos","epsneg","accrualpos","accrualneg")
yvar_loc1 = which(startsWith(colnamesdataset,'F1'))
yvar_loc2 = which(startsWith(colnamesdataset,'F2')) 

yvar1 = colnamesdataset[yvar_loc1]
yvar2 = colnamesdataset[yvar_loc2] 
####all candidate variables for outcome variable 
yvar = c(yvar1,yvar2) 

####storing all yvar and inputvar 
write.table(yvar,'yvar.txt',row.names = F)
write.table(inputvar,'inputvar.txt',row.names = F)

######inputvar2 : all candidates input variables 
inputvar2 = colnamesdataset[-c(1,2,3,21,110,yvar_loc1,yvar_loc2)]
write.table(inputvar2,'inputvar2.txt',row.names = F)



#####use yvar and inputvar2 


#####For F1 (1 year ahead forecasting)


###All the enddates
end_dates = seq(1989,2018,1)






####Creat files for resutls 
MLonly1 = matrix(data = NA,nrow = length(end_dates))
MLman1 = matrix(data = NA,nrow = length(end_dates))
error_MLonly1 =  matrix(data = NA,nrow = length(end_dates))
error_MLman1 =  matrix(data = NA,nrow = length(end_dates))

####For each iteration, I predict earnings and bias for each end_dates
####There are four settings: MLonly, MLman = ML+man, error_MLonly(for bias),error_MLman

for(m in seq(length(end_dates))){   
  end_date = end_dates[m]   
  ###choose previous 10 years data
  indf1 = ((end_date-10) <= dataset_pershare['fyear']) & (dataset_pershare['fyear'] < end_date)

  dataset_f1 = dataset_pershare[indf1,] 
  dataset_f1 = cv(dataset_f1,0.9)

  
  ####For F1  
  ######split data into training and validation dataset
  training_all1 = dataset_f1[dataset_f1['MODELING_GROUP']==0,]
  validation_all1 =  dataset_f1[dataset_f1['MODELING_GROUP']==1,]
  
  ###### split both training and validation into candidate outcome variables(y) and candidate input variables
  y_training1 = training_all1[,yvar]
  
  x_training1 = training_all1[,inputvar2]   
  
  y_vali1 = validation_all1[,yvar]
  
  x_vali1 = validation_all1[,inputvar2] 
  
  
  ###number of predictors 
  nump = ncol(x_training1)  
  ###Hyperparameter candidates: Number of variables randomly sampled as candidates at each split
  lamv= seq(10,nump,by=50)  
  ###number of trees to grow 
  ne = 100  
  #####Maximum nodes possible 
  lamc = c(4,8,16,32)  
  ### matrixs for storing result of tuning 
  
  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  

  for (n1 in seq(length(lamv))){ 
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
      nc = lamc[n2]  
      ####yvar to use is F1eps_IBES_trim
      clf = randomForest(x_training1[,inputvar],y_training1$F1eps_IBES_trim,ntree = ne,mtry = nf,maxnodes = nc)
      #####prediction for validation
      yhatbig1 = predict(clf,newdata = x_vali1[,inputvar],type = 'response')

      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      #### calculation of beat ratio for validation set
      r1[n1,n2]=sum(abs(yhatbig1-y_vali1$F1eps_IBES)<abs(x_vali1$ana_forecast_f1eps-y_vali1$F1eps_IBES))/nrow(y_vali1)
      
      #####Prediction for the whole dataset 
      yhatbig1= predict(clf,newdata = dataset_f1[,inputvar],type = 'response')
      r2[n1,n2]= sum(abs(yhatbig1-dataset_f1$F1eps_IBES)<abs(dataset_f1$ana_forecast_f1eps-dataset_f1$F1eps_IBES))/nrow(dataset_f1)
    }
    
  }
  r1 = as.vector(r1)
  r2 = as.vector(r2)
  
  MLonly1[m]= r2[which.max(r1)]
  
  
  #####ML + man 
  
  ####Inputvariables for ML+man seeting  
  inputvarman1 = c(inputvar, "fy1_con_ana_n","fy1_con_ana_min","fy1_con_ana_max","fy1_con_ana_mean","fy1_con_ana_std" )
  
  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  
  
  for (n1 in seq(length(lamv))){ 
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
      nc = lamc[n2]  
      ####yvar to use is F1eps_IBES_trim
      clf = randomForest(x_training1[,inputvarman1],y_training1$F1eps_IBES_trim,ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = x_vali1[,inputvarman1],type = 'response')
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2]=sum(abs(yhatbig1-y_vali1$F1eps_IBES)<abs(x_vali1$ana_forecast_f1eps-y_vali1$F1eps_IBES))/nrow(y_vali1)
      
      
      yhatbig1= predict(clf,newdata = x_training1[,inputvarman1],type = 'response')
      r2[n1,n2]= sum(abs(yhatbig1-y_training1$F1eps_IBES)<abs(x_training1$ana_forecast_f1eps-y_training1$F1eps_IBES))/nrow(y_training1)
      
    }
  }
  r1 = as.vector(r1)
  r2 = as.vector(r2)

  MLman1[m]= r2[which.max(r1)]



########predict analyst error  ML only 

  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  

  for (n1 in seq(length(lamv))){ 
    
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
    nc = lamc[n2]  
    ####yvar to use is F1eps_IBES_trim
    clf = randomForest(x_training1[,inputvar],(y_training1$F1eps_IBES_trim-x_training1$ana_forecast_f1eps),ntree = ne,mtry = nf,maxnodes = nc)
    
    yhatbig1 = predict(clf,newdata = x_vali1[,inputvar],type = 'response') + x_vali1$ana_forecast_f1eps
    
    
    #######beat ratio  
    ###Analyst's forecast 
    ###ana_forecast_f1eps 
    ###ana_forecast_f2eps 
    
    r1[n1,n2]=mean(abs(yhatbig1-y_vali1$F1eps_IBES)<abs(x_vali1$ana_forecast_f1eps-y_vali1$F1eps_IBES))
    
    
    yhatbig1= predict(clf,newdata = dataset_f1[,inputvar],type = 'response')+ dataset_f1$ana_forecast_f1eps
    r2[n1,n2]= sum(abs(yhatbig1-dataset_f1$F1eps_IBES)<abs(dataset_f1$ana_forecast_f1eps-dataset_f1$F1eps_IBES))/nrow(dataset_f1)
    
  }
}


  r1 = as.vector(r1)
  r2 = as.vector(r2)

  error_MLonly1[m]= r2[which.max(r1)]


########predict analyst error  ML+man only 

  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  

  for (n1 in seq(length(lamv))){ 
  nf=lamv[n1]
  for (n2 in seq(length(lamc)) ){ 
    nc = lamc[n2]  
    ####yvar to use is F1eps_IBES_trim
    clf = randomForest(x_training1[,inputvarman1],(y_training1$F1eps_IBES_trim-x_training1$ana_forecast_f1eps),ntree = ne,mtry = nf,maxnodes = nc)
    
    yhatbig1 = predict(clf,newdata = x_vali1[,inputvarman1],type = 'response') +  x_vali1$ana_forecast_f1eps
    
    
    #######beat ratio  
    ###Analyst's forecast 
    ###ana_forecast_f1eps 
    ###ana_forecast_f2eps 
    
    r1[n1,n2]=sum(abs(yhatbig1-y_vali1$F1eps_IBES)<abs(x_vali1$ana_forecast_f1eps-y_vali1$F1eps_IBES))/nrow(y_vali1)
    
    
    yhatbig1= predict(clf,newdata = dataset_f1[,inputvarman1],type = 'response')+ dataset_f1$ana_forecast_f1eps
    r2[n1,n2]= sum(abs(yhatbig1-dataset_f1$F1eps_IBES)<abs(dataset_f1$ana_forecast_f1eps-dataset_f1$F1eps_IBES))/nrow(dataset_f1)
    
  }
}


  r1 = as.vector(r1)
  r2 = as.vector(r2)

  error_MLman1[m]= r2[which.max(r1)]

}





############For two year ahead forecast ################# 

#####For F2 



end_dates = seq(1989,2018,1)






####Creat files for resutls 
MLonly2 = matrix(data = NA,nrow = length(end_dates))
MLman2 = matrix(data = NA,nrow = length(end_dates))
error_MLonly2 =  matrix(data = NA,nrow = length(end_dates))
error_MLman2 =  matrix(data = NA,nrow = length(end_dates))



for(m in seq(length(end_dates))){ 
  end_date = end_dates[m]  
  indf2 = ((end_date-11) <= dataset_pershare['fyear']) & (dataset_pershare['fyear'] < (end_date-1))
  dataset_f2 = dataset_pershare[indf2,]
  dataset_f2 =cv(dataset_f2,0.9) 
  
  
  ####For F2 
  training_all2 = dataset_f2[dataset_f2['MODELING_GROUP']==0,]
  validation_all2 =  dataset_f2[dataset_f2['MODELING_GROUP']==1,]
  
  
  y_training2 = training_all2[,yvar]
  
  x_training2 = training_all2[,inputvar2]   
  
  y_vali2 = validation_all2[,yvar]
  
  x_vali2 = validation_all2[,inputvar2] 
  
  
  ###number of predictors 
  nump = ncol(x_training2) 
  lamv= seq(10,nump,by=50) 
  ne = 100  
  #####Maximum nodes possible 
  lamc = c(4,8,16,32) 
  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  
  
  for (n1 in seq(length(lamv))){ 
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
      nc = lamc[n2]  
      ####yvar to use is F1eps_IBES_trim
      clf = randomForest(x_training2[,inputvar],y_training2$F2eps_IBES_trim,ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = x_vali2[,inputvar],type = 'response')
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2] = sum(abs(yhatbig1-y_vali2$F2eps_IBES)<abs(x_vali2$ana_forecast_f2eps-y_vali2$F2eps_IBES))/nrow(y_vali2)
      
      
      yhatbig1= predict(clf,newdata = dataset_f2[,inputvar],type = 'response')
      r2[n1,n2]= sum(abs(yhatbig1-dataset_f2$F1eps_IBES)<abs(dataset_f2$ana_forecast_f2eps-dataset_f2$F2eps_IBES))/nrow(dataset_f2)
    }
    
  }
  r1 = as.vector(r1)
  r2 = as.vector(r2)
  
  MLonly2[m]= r2[which.max(r1)]
  
  
  #####ML + man 
  
  
  inputvarman2 = c(inputvar, "fy2_con_ana_n","fy2_con_ana_min","fy2_con_ana_max","fy2_con_ana_mean","fy2_con_ana_std" )
  
  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  
  
  for (n1 in seq(length(lamv))){ 
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
      nc = lamc[n2]  
      ####yvar to use is F1eps_IBES_trim
      clf = randomForest(x_training2[,inputvarman2],y_training2$F2eps_IBES_trim,ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = x_vali2[,inputvarman2],type = 'response')
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2]=sum(abs(yhatbig1-y_vali2$F2eps_IBES)<abs(x_vali2$ana_forecast_f2eps-y_vali2$F2eps_IBES))/nrow(y_vali2)
      
      
      yhatbig1= predict(clf,newdata = dataset_f2[,inputvarman2],type = 'response')
      r2[n1,n2]= sum(abs(yhatbig1-dataset_f2$F2eps_IBES)<abs(dataset_f2$ana_forecast_f2eps-dataset_f2$F2eps_IBES))/nrow(dataset_f2)
      
    }
  }
  r1 = as.vector(r1)
  r2 = as.vector(r2)
  
  MLman2[m]= r2[which.max(r1)]
  
  
  
  ########predict analyst error  ML only 
  
  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  
  
  for (n1 in seq(length(lamv))){ 
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
      nc = lamc[n2]  
      ####yvar to use is F1eps_IBES_trim
      clf = randomForest(x_training2[,inputvar],(y_training2$F1eps_IBES_trim-x_training2$ana_forecast_f2eps),ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = x_vali2[,inputvar],type = 'response') + x_vali2$ana_forecast_f2eps
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2]=mean(abs(yhatbig1-y_vali2$F2eps_IBES)<abs(x_vali2$ana_forecast_f2eps-y_vali2$F2eps_IBES))
      
      
      yhatbig1= predict(clf,newdata = dataset_f2[,inputvar],type = 'response')+ dataset_f2$ana_forecast_f2eps
      r2[n1,n2]= sum(abs(yhatbig1-dataset_f2$F2eps_IBES)<abs(dataset_f2$ana_forecast_f2eps-dataset_f2$F2eps_IBES))/nrow(dataset_f2)
      
    }
  }
  
  
  r1 = as.vector(r1)
  r2 = as.vector(r2)
  
  error_MLonly2[m]= r2[which.max(r1)]
  
  
  ########predict analyst error  ML+man only 
  
  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  
  
  for (n1 in seq(length(lamv))){ 
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
      nc = lamc[n2]  
      ####yvar to use is F1eps_IBES_trim
      clf = randomForest(x_training2[,inputvarman2],(y_training2$F2eps_IBES_trim-x_training2$ana_forecast_f2eps),ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = x_vali2[,inputvarman2],type = 'response') +  x_vali2$ana_forecast_f2eps
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2]=sum(abs(yhatbig1-y_vali2$F2eps_IBES)<abs(x_vali2$ana_forecast_f2eps-y_vali2$F2eps_IBES))/nrow(y_vali2)
      
      
      yhatbig1= predict(clf,newdata = dataset_f2[,inputvarman2],type = 'response')+ dataset_f2$ana_forecast_f2eps
      r2[n1,n2]= sum(abs(yhatbig1-dataset_f2$F2eps_IBES)<abs(dataset_f2$ana_forecast_f2eps-dataset_f2$F2eps_IBES))/nrow(dataset_f2)
      
    }
  }
  
  
  r1 = as.vector(r1)
  r2 = as.vector(r2)
  
  error_MLman2[m]= r2[which.max(r1)]
  
}









