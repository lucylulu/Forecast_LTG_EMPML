
"""
This version: September 4th, 2021

Purpose: Replication of the Earnings-part-2
This version modified some of the mistakes of previous version in calculating beat ratio, computing missing value.  


@author: Xia Zou
"""  




####remove everything
rm(list = ls()) 
library(haven)
library(randomForest) 
library(dplyr)
####path of data
# pathdta = '/Users/xiazou/Desktop/Tinbergen_Study/block5/Forecasting_LTG/Replication/OneDrive-2021-08-19'
dataset_at_pershare = read_dta('dataset_at_pershare.dta')
setwd(pathdta)

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
  data_id[data_id$wookie > percentage,'MODELING_GROUP'] = 1
  data_out = merge(dataset,data_id,by='gvkey')
  return(data_out)
}




####Rough missing value imputation??? replace all the missing values with 0

# miss_with0= function(x){
#   x[is.na(x)]=0
#   return(x)
# }
#
# for(i in seq(ncol(dataset_pershare))){
#   dataset_pershare[,i]= miss_with0(dataset_pershare[,i])
# }
#
#
# ######Dealing with missing value

#####################create indicators that indicate whether the outcome variables is NA

dataset_pershare[,'is_naf1'] = is.na(dataset_pershare$F1eps_IBES)

dataset_pershare[,'is_naf2'] = is.na(dataset_pershare$F2eps_IBES)


#
###############missing data for characteristics data  ***
##First step: delete dates that has all variables are na
all_na_ind <- function(x){
  n_x = length(x)
  n_na = sum(is.na(x))
  #ind = 0 means vector x is not all na value
  ind = 0
  if(n_x == n_na){
    ind = 1
  }
  return(ind)
}

na_ind_dtach = dataset_pershare %>% group_by(fyear) %>% summarise_all(all_na_ind)



na_ind_dtach = as.data.frame(na_ind_dtach)


######Replace dates with all nas with 0

all_na = as.data.frame(colSums(na_ind_dtach[,-1]))

#variables with all na
var_all_na = colnames(na_ind_dtach)[which(all_na$`colSums(na_ind_dtach[, -1])` != 0)+1 ]
#
for (var_na in var_all_na){ 
  id_na = which(colnames(dataset_pershare) == var_na)
  date_na = na_ind_dtach[na_ind_dtach[[id_na]]==1,'fyear'] 
  dataset_pershare[which(dataset_pershare$fyear %in% date_na),][[id_na]] = 0
}

#dta_sub_ch_1 = subset(dta_sub_ch, DATE %in% na_ind_dtach[rowSums(as.data.frame(na_ind_dtach[,-1]))==0,]$DATE)




impute_median <- function(x){
  ind_na = is.na(x) 
  x[ind_na] = median(x[!ind_na]) 
  return(as.numeric(x))
}

dataset_pershare2 = dataset_pershare[,-2] %>% group_by(fyear)%>% mutate_all(impute_median)
dataset_pershare = dataset_pershare2
rm(dataset_pershare2)








# dataset_price_used =  read_dta('dataset_price_used.dta')
# dataset_unscaled = read_dta('dataset_unscaled.dta')
###extracting input variables for training 

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
####First try 1989 to 2000
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
  dataset_enddate = dataset_pershare[dataset_pershare$fyear==end_date,]
  
  
  
  
  ####For F1  
  ######split data into training and validation dataset
  training_all1 = dataset_f1[dataset_f1['MODELING_GROUP']==0,]
  validation_all1 =  dataset_f1[dataset_f1['MODELING_GROUP']==1,]
  
  ###### split both training and validation into candidate outcome variables(y) and candidate input variables
  # y_training1 = training_all1[,yvar]
  # 
  # x_training1 = training_all1[,inputvar2]   
  # 
  # y_vali1 = validation_all1[,yvar]
  # 
  # x_vali1 = validation_all1[,inputvar2] 
  # 
  
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
      clf = randomForest(training_all1[,inputvar],training_all1$F1eps_IBES_trim,ntree = ne,mtry = nf,maxnodes = nc)
      #####prediction for validation
      yhatbig1 = predict(clf,newdata = validation_all1[,inputvar],type = 'response')

      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      #### calculation of beat ratio for validation set
      r1[n1,n2]=mean(abs(yhatbig1-validation_all1$F1eps_IBES)<abs(validation_all1$ana_forecast_f1eps-validation_all1$F1eps_IBES))
      
      # #####Prediction for the outsample 
      # yhatbig1= predict(clf,newdata = dataset_f1[,inputvar],type = 'response')
      # r2[n1,n2]= sum(abs(yhatbig1-dataset_f1$F1eps_IBES)<abs(dataset_f1$ana_forecast_f1eps-dataset_f1$F1eps_IBES))/nrow(dataset_f1)
    }
    
  }
  #r2 = as.vector(r2) 
  max_loc = which(r1 == max(r1), arr.ind = TRUE)
  clf_best = randomForest(dataset_f1[,inputvar],dataset_f1$F1eps_IBES_trim,ntree = ne,mtry = max_loc[1],maxnodes = max_loc[2])
  
  #####Prediction for the outsample  
  newdataset = dataset_enddate[dataset_enddate$is_naf1 ==0,]
  yhatbig1= predict(clf_best,newdata = newdataset[,inputvar],type = 'response')
  
  
  
  MLonly1[m]= mean(abs(yhatbig1-newdataset$F1eps_IBES)<abs(newdataset$ana_forecast_f1eps-newdataset$F1eps_IBES))

  
  
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
      clf = randomForest(training_all1[,inputvarman1],training_all1$F1eps_IBES_trim,ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = validation_all1[,inputvarman1],type = 'response')
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2]=mean(abs(yhatbig1-validation_all1$F1eps_IBES)<abs(validation_all1$ana_forecast_f1eps-validation_all1$F1eps_IBES))
      
      
      # yhatbig1= predict(clf,newdata = x_training1[,inputvarman1],type = 'response')
      # r2[n1,n2]= sum(abs(yhatbig1-y_training1$F1eps_IBES)<abs(x_training1$ana_forecast_f1eps-y_training1$F1eps_IBES))/nrow(y_training1)
      # 
    }
  }
  
  max_loc = which(r1 == max(r1), arr.ind = TRUE)
  clf_best = randomForest(dataset_f1[,inputvarman1],dataset_f1$F1eps_IBES_trim,ntree = ne,mtry = max_loc[1],maxnodes = max_loc[2])
  
  #####Prediction for the outsample  
  newdataset = dataset_enddate[dataset_enddate$is_naf1 ==0,]
  
  yhatbig1= predict(clf_best,newdata = newdataset[,inputvarman1],type = 'response')
  
  
  
  MLman1[m]= mean(abs(yhatbig1-newdataset$F1eps_IBES)<abs(newdataset$ana_forecast_f1eps-newdataset$F1eps_IBES))
  
  
  



########predict analyst error  ML only 

  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  

  for (n1 in seq(length(lamv))){ 
    
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
    nc = lamc[n2]  
    ####yvar to use is F1eps_IBES_trim
    clf = randomForest(training_all1[,inputvar],(training_all1$F1eps_IBES_trim-training_all1$ana_forecast_f1eps),ntree = ne,mtry = nf,maxnodes = nc)
    
    yhatbig1 = predict(clf,newdata = validation_all1[,inputvar],type = 'response') + validation_all1$ana_forecast_f1eps
    
    
    #######beat ratio  
    ###Analyst's forecast 
    ###ana_forecast_f1eps 
    ###ana_forecast_f2eps 
    
    r1[n1,n2]=mean(abs(yhatbig1-validation_all1$F1eps_IBES)<abs(validation_all1$ana_forecast_f1eps-validation_all1$F1eps_IBES))
    
    
    # yhatbig1= predict(clf,newdata = dataset_f1[,inputvar],type = 'response')+ dataset_f1$ana_forecast_f1eps
    # r2[n1,n2]= sum(abs(yhatbig1-dataset_f1$F1eps_IBES)<abs(dataset_f1$ana_forecast_f1eps-dataset_f1$F1eps_IBES))/nrow(dataset_f1)
    # 
  }
}
  max_loc = which(r1 == max(r1), arr.ind = TRUE)
  clf_best = randomForest(dataset_f1[,inputvar],(dataset_f1$F1eps_IBES_trim-dataset_f1$ana_forecast_f1eps),ntree = ne,mtry = max_loc[1],maxnodes = max_loc[2])
  
  #####Prediction for the outsample  
  newdataset = dataset_enddate[dataset_enddate$is_naf1 ==0,]
  
  yhatbig1= predict(clf_best,newdata = newdataset[,inputvar],type = 'response')+ newdataset$ana_forecast_f1eps
  
  
  
  
  
  
  error_MLonly1[m]= mean(abs(yhatbig1-newdataset$F1eps_IBES)<abs(newdataset$ana_forecast_f1eps-newdataset$F1eps_IBES))



########predict analyst error  ML+man only 

  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  

  for (n1 in seq(length(lamv))){ 
  nf=lamv[n1]
  for (n2 in seq(length(lamc)) ){ 
    nc = lamc[n2]  
    ####yvar to use is F1eps_IBES_trim
    clf = randomForest(training_all1[,inputvarman1],(training_all1$F1eps_IBES_trim-training_all1$ana_forecast_f1eps),ntree = ne,mtry = nf,maxnodes = nc)
    
    yhatbig1 = predict(clf,newdata = validation_all1[,inputvarman1],type = 'response') +  validation_all1$ana_forecast_f1eps
    
    
    #######beat ratio  
    ###Analyst's forecast 
    ###ana_forecast_f1eps 
    ###ana_forecast_f2eps 
    
    r1[n1,n2]=mean(abs(yhatbig1-validation_all1$F1eps_IBES)<abs(validation_all1$ana_forecast_f1eps-validation_all1$F1eps_IBES))
    
    # 
    # yhatbig1= predict(clf,newdata = dataset_f1[,inputvarman1],type = 'response')+ dataset_f1$ana_forecast_f1eps
    # r2[n1,n2]= sum(abs(yhatbig1-dataset_f1$F1eps_IBES)<abs(dataset_f1$ana_forecast_f1eps-dataset_f1$F1eps_IBES))/nrow(dataset_f1)
    # 
  }
}


  max_loc = which(r1 == max(r1), arr.ind = TRUE)
  clf_best = randomForest(dataset_f1[,inputvarman1],(dataset_f1$F1eps_IBES_trim-dataset_f1$ana_forecast_f1eps),ntree = ne,mtry = max_loc[1],maxnodes = max_loc[2])
  
  #####Prediction for the outsample 
  newdataset = dataset_enddate[dataset_enddate$is_naf1 ==0,]
  
  yhatbig1= predict(clf_best,newdata = newdataset[,inputvarman1],type = 'response')+ newdataset$ana_forecast_f1eps
  
  
  
  
  
  
  error_MLman1[m]= mean(abs(yhatbig1-newdataset$F1eps_IBES)<abs(newdataset$ana_forecast_f1eps-newdataset$F1eps_IBES))

}





############For two year ahead forecast ################# 

#####For F2 



end_dates = seq(1989,2018,1)






####Creat files for resutls 
MLonly2 = matrix(data = NA,nrow = length(end_dates))
MLman2 = matrix(data = NA,nrow = length(end_dates))
error_MLonly2 =  matrix(data = NA,nrow = length(end_dates))
error_MLman2 =  matrix(data = NA,nrow = length(end_dates))


m=1
for(m in seq(length(end_dates))){ 
  end_date = end_dates[m]  
  indf2 = ((end_date-11) <= dataset_pershare['fyear']) & (dataset_pershare['fyear'] < (end_date-1))
  dataset_f2 = dataset_pershare[indf2,]
  dataset_f2 =cv(dataset_f2,0.9)  
  dataset_enddate = dataset_pershare[dataset_pershare$fyear==end_date,]
  
  
  ####For F2 
  training_all2 = dataset_f2[dataset_f2['MODELING_GROUP']==0,]
  validation_all2 =  dataset_f2[dataset_f2['MODELING_GROUP']==1,]
  
  # 
  # y_training2 = training_all2[,yvar]
  # 
  # x_training2 = training_all2[,inputvar2]   
  # 
  # y_vali2 = validation_all2[,yvar]
  # 
  # x_vali2 = validation_all2[,inputvar2] 
  # 
  
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
      clf = randomForest(training_all2[,inputvar],training_all2$F2eps_IBES_trim,ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = validation_all2[,inputvar],type = 'response')
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2] = mean(abs(yhatbig1-validation_all2$F2eps_IBES)<abs(validation_all2$ana_forecast_f2eps-validation_all2$F2eps_IBES))
      
      
      # yhatbig1= predict(clf,newdata = dataset_f2[,inputvar],type = 'response')
      # r2[n1,n2]= sum(abs(yhatbig1-dataset_f2$F1eps_IBES)<abs(dataset_f2$ana_forecast_f2eps-dataset_f2$F2eps_IBES))/nrow(dataset_f2)
    }
    
  }
  
  max_loc = which(r1 == max(r1), arr.ind = TRUE)
  clf_best = randomForest(dataset_f2[,inputvarman1],dataset_f2$F2eps_IBES_trim,ntree = ne,mtry = max_loc[1],maxnodes = max_loc[2])
  
  #####Prediction for the outsample 
  newdataset = dataset_enddate[dataset_enddate$is_naf2 ==0,]
  
  yhatbig1= predict(clf_best,newdata = newdataset[,inputvarman1],type = 'response')
  
  
  
  
  
  
  MLonly2[m]= mean(abs(yhatbig1-newdataset$F2eps_IBES)<abs(newdataset$ana_forecast_f2eps-newdataset$F2eps_IBES))
  

  

  
  #####ML + man 
  
  
  inputvarman2 = c(inputvar, "fy2_con_ana_n","fy2_con_ana_min","fy2_con_ana_max","fy2_con_ana_mean","fy2_con_ana_std" )
  
  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  
  
  for (n1 in seq(length(lamv))){ 
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
      nc = lamc[n2]  
      ####yvar to use is F1eps_IBES_trim
      clf = randomForest(training_all2[,inputvarman2],training_all2$F2eps_IBES_trim,ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = validation_all2[,inputvarman2],type = 'response')
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2]=mean(abs(yhatbig1-validation_all2$F2eps_IBES)<abs(validation_all2$ana_forecast_f2eps-validation_all2$F2eps_IBES))
      
      
      # yhatbig1= predict(clf,newdata = dataset_f2[,inputvarman2],type = 'response')
      # r2[n1,n2]= sum(abs(yhatbig1-dataset_f2$F2eps_IBES)<abs(dataset_f2$ana_forecast_f2eps-dataset_f2$F2eps_IBES))/nrow(dataset_f2)
      # 
    }
  }
  max_loc = which(r1 == max(r1), arr.ind = TRUE)
  clf_best = randomForest(dataset_f2[,inputvarman2],dataset_f2$F2eps_IBES_trim,ntree = ne,mtry = max_loc[1],maxnodes = max_loc[2])
  
  #####Prediction for the outsample 
  newdataset = dataset_enddate[dataset_enddate$is_naf2 ==0,]
  
  yhatbig1= predict(clf_best,newdata = newdataset[,inputvarman2],type = 'response')
  
  
  
  
  
  
  MLman2[m]= mean(abs(yhatbig1-newdataset$F2eps_IBES)<abs(newdataset$ana_forecast_f2eps-newdataset$F2eps_IBES))
  
  
  
  
  ########predict analyst error  ML only 
  
  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  
  
  for (n1 in seq(length(lamv))){ 
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
      nc = lamc[n2]  
      ####yvar to use is F1eps_IBES_trim
      clf = randomForest(training_all2[,inputvar],(training_all2$F2eps_IBES_trim-training_all2$ana_forecast_f2eps),ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = validation_all2[,inputvar],type = 'response') + validation_all2$ana_forecast_f2eps
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2]=mean(abs(yhatbig1-validation_all2$F2eps_IBES)<abs(validation_all2$ana_forecast_f2eps-validation_all2$F2eps_IBES))
      
      
      # yhatbig1= predict(clf,newdata = dataset_f2[,inputvar],type = 'response')+ dataset_f2$ana_forecast_f2eps
      # r2[n1,n2]= sum(abs(yhatbig1-dataset_f2$F2eps_IBES)<abs(dataset_f2$ana_forecast_f2eps-dataset_f2$F2eps_IBES))/nrow(dataset_f2)
      # 
    }
  }
  
  max_loc = which(r1 == max(r1), arr.ind = TRUE)
  clf_best = randomForest(dataset_f2[,inputvarman1],(dataset_f2$F2eps_IBES_trim-dataset_f2$ana_forecast_f2eps),ntree = ne,mtry = max_loc[1],maxnodes = max_loc[2])
  
  #####Prediction for the outsample 
  newdataset = dataset_enddate[dataset_enddate$is_naf2 ==0,]
  
  yhatbig1= predict(clf_best,newdata = newdataset[,inputvarman1],type = 'response')+ newdataset$ana_forecast_f2eps
  
  
  
  
  
  
  error_MLonly2[m]= mean(abs(yhatbig1-newdataset$F2eps_IBES)<abs(newdataset$ana_forecast_f2eps-newdataset$F2eps_IBES))
  
  
  
  
  ########predict analyst error  ML+man only 
  
  r1 = matrix(data=NA,length(lamv),length(lamc)) 
  r2 = matrix(data=NA,length(lamv),length(lamc))  
  
  for (n1 in seq(length(lamv))){ 
    nf=lamv[n1]
    for (n2 in seq(length(lamc)) ){ 
      nc = lamc[n2]  
      ####yvar to use is F1eps_IBES_trim
      clf = randomForest(training_all2[,inputvarman2],(training_all2$F2eps_IBES_trim-training_all2$ana_forecast_f2eps),ntree = ne,mtry = nf,maxnodes = nc)
      
      yhatbig1 = predict(clf,newdata = validation_all2[,inputvarman2],type = 'response') +  validation_all2$ana_forecast_f2eps
      
      
      #######beat ratio  
      ###Analyst's forecast 
      ###ana_forecast_f1eps 
      ###ana_forecast_f2eps 
      
      r1[n1,n2]=mean(abs(yhatbig1-validation_all2$F2eps_IBES)<abs(validation_all2$ana_forecast_f2eps-validation_all2$F2eps_IBES))
      
      # 
      # yhatbig1= predict(clf,newdata = dataset_f2[,inputvarman2],type = 'response')+ dataset_f2$ana_forecast_f2eps
      # r2[n1,n2]= sum(abs(yhatbig1-dataset_f2$F2eps_IBES)<abs(dataset_f2$ana_forecast_f2eps-dataset_f2$F2eps_IBES))/nrow(dataset_f2)
      # 
    }
  }
  
  max_loc = which(r1 == max(r1), arr.ind = TRUE)
  clf_best = randomForest(dataset_f2[,inputvarman2],(dataset_f2$F2eps_IBES_trim-dataset_f2$ana_forecast_f2eps),ntree = ne,mtry = max_loc[1],maxnodes = max_loc[2])
  
  #####Prediction for the outsample 
  newdataset = dataset_enddate[dataset_enddate$is_naf2 ==0,]
  
  yhatbig1= predict(clf_best,newdata = newdataset[,inputvarman2],type = 'response')+ newdataset$ana_forecast_f2eps
  
  
  
  
  
  
  error_MLman2[m]= mean(abs(yhatbig1-newdataset$F2eps_IBES)<abs(newdataset$ana_forecast_f2eps-newdataset$F2eps_IBES))
  
  
  
  
}









