
"""
This version: September 5th, 2021

Purpose: Replication of the Earnings-part-2
This version speed up the code. 


@author: Xia Zou
"""  




####remove everything
rm(list = ls()) 
library(haven)
library(randomForest) 
library(dplyr) 
library(h2o) 
library(ggplot2) 
library(reshape2)
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

######Create bias variable for f1 and f2 
dataset_pershare[,'error_f1']= dataset_pershare$F1eps_IBES_trim - dataset_pershare$ana_forecast_f1eps
dataset_pershare[,'error_f2']= dataset_pershare$F2eps_IBES_trim - dataset_pershare$ana_forecast_f2eps







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
####Or should try 1988 to 2017 ?????????
end_dates = seq(1989,2018,1)


h2o.init(nthreads = 4,max_mem_size = '9g')


####Creat files for resutls 
MLonly1 = matrix(data = NA,nrow = length(end_dates))
MLman1 = matrix(data = NA,nrow = length(end_dates))
error_MLonly1 =  matrix(data = NA,nrow = length(end_dates))
error_MLman1 =  matrix(data = NA,nrow = length(end_dates))

####For each iteration, I predict earnings and bias for each end_dates
####There are four settings: MLonly, MLman = ML+man, error_MLonly(for bias),error_MLman
m=1
for(m in seq(length(end_dates))){   
  
  end_date = end_dates[m]   
  ###choose previous 10 years data
  indf1 = ((end_date-10) <= dataset_pershare['fyear']) & (dataset_pershare['fyear'] < end_date)

  dataset_f1 = dataset_pershare[indf1,] 
  dataset_f1 = cv(dataset_f1,0.9)
  dataset_enddate = as.h2o(dataset_pershare[dataset_pershare$fyear==end_date,])
  
  
  
  
  ####For F1  
  ######split data into training and validation dataset
  training_all1 = as.h2o(dataset_f1[dataset_f1['MODELING_GROUP']==0,])
  validation_all1 =  as.h2o(dataset_f1[dataset_f1['MODELING_GROUP']==1,])
  newdataset = dataset_enddate[dataset_enddate$is_naf1 ==0,]
  

  ###ML only   
  clf = h2o.randomForest(x= inputvar,y='F1eps_IBES_trim',training_frame=(training_all1),validation_frame = (validation_all1))
    
  yhatbig1= predict(clf,newdata = newdataset[,inputvar],type = 'response')
  
  MLonly1[m]= mean(abs(yhatbig1-newdataset$F1eps_IBES)<abs(newdataset$ana_forecast_f1eps-newdataset$F1eps_IBES))

  
  
  #####ML + man 
  
  ####Inputvariables for ML+man seeting  
  inputvarman1 = c(inputvar, "fy1_con_ana_n","fy1_con_ana_min","fy1_con_ana_max","fy1_con_ana_mean","fy1_con_ana_std" )
  
  clf =  h2o.randomForest(x= inputvarman1,y='F1eps_IBES_trim',training_frame=(training_all1),validation_frame = (validation_all1))
  yhatbig1= predict(clf,newdata = newdataset[,inputvarman1],type = 'response')
  
  
  
  MLman1[m]= mean(abs(yhatbig1-newdataset$F1eps_IBES)<abs(newdataset$ana_forecast_f1eps-newdataset$F1eps_IBES))
  
  
  



########predict analyst error  ML only 

 
  clf =  h2o.randomForest(x= inputvar,y='error_f1',training_frame=training_all1,validation_frame = validation_all1)
  
  yhatbig1= predict(clf,newdata = newdataset[,inputvar],type = 'response')+ newdataset$ana_forecast_f1eps
  
  
  
  
  
  
  error_MLonly1[m]= mean(abs(yhatbig1-newdataset$F1eps_IBES)<abs(newdataset$ana_forecast_f1eps-newdataset$F1eps_IBES))



########predict analyst error  ML+man only 
  clf = h2o.randomForest(x= inputvarman1,y='error_f1',training_frame=training_all1,validation_frame = validation_all1)

  yhatbig1= predict(clf,newdata = newdataset[,inputvarman1],type = 'response')+ newdataset$ana_forecast_f1eps
  error_MLman1[m]= mean(abs(yhatbig1-newdataset$F1eps_IBES)<abs(newdataset$ana_forecast_f1eps-newdataset$F1eps_IBES))
  
  h2o.removeAll()
  
  }


h2o.shutdown() 

dataplot=as.data.frame(cbind(end_dates,MLonly1,MLman1,error_MLonly1,error_MLman1))
dataplot = dataplot[-nrow(dataplot),]
colnames(dataplot) = c('end_dates','MLonly1','MLman1','error_MLonly1','error_MLman1')

dataplot2 =melt(dataplot, id.vars = c('end_dates'))







ggplot(data = dataplot2 ,aes(x=end_dates,y=value,group = variable,color= variable))+geom_line()+ggtitle('1 year ahead forecasting')



############For two year ahead forecast ################# 

#####For F2 



###All the enddates 
####First try 1988 to 2017
end_dates = seq(1989,2018,1)
h2o.init(nthreads = 4,max_mem_size = '9g')




####Creat files for resutls 
MLonly2 = matrix(data = NA,nrow = length(end_dates))
MLman2 = matrix(data = NA,nrow = length(end_dates))
error_MLonly2 =  matrix(data = NA,nrow = length(end_dates))
error_MLman2=  matrix(data = NA,nrow = length(end_dates))

####For each iteration, I predict earnings and bias for each end_dates
####There are four settings: MLonly, MLman = ML+man, error_MLonly(for bias),error_MLman


for(m in seq(length(end_dates))){   
  
  end_date = end_dates[m]   
  ###choose previous 10 years data
  indf2 = ((end_date-11) <= dataset_pershare['fyear']) & (dataset_pershare['fyear'] < (end_date-1))
  
  dataset_f2 = dataset_pershare[indf2,] 
  dataset_f2 = cv(dataset_f2,0.9)
  dataset_enddate = as.h2o(dataset_pershare[dataset_pershare$fyear==end_date,])
  
  
  
  
  ####For F2 
  ######split data into training and validation dataset
  training_all2 = as.h2o(dataset_f2[dataset_f2['MODELING_GROUP']==0,])
  validation_all2 =  as.h2o(dataset_f2[dataset_f2['MODELING_GROUP']==1,])
  newdataset = dataset_enddate[dataset_enddate$is_naf2 ==0,]
  
  
  ###ML only   
  clf = h2o.randomForest(x= inputvar,y='F2eps_IBES_trim',training_frame=(training_all2),validation_frame = (validation_all2))
  
  yhatbig1= predict(clf,newdata = newdataset[,inputvar],type = 'response')
  
  MLonly2[m]= mean(abs(yhatbig1-newdataset$F2eps_IBES)<abs(newdataset$ana_forecast_f2eps-newdataset$F2eps_IBES))
  
  
  
  #####ML + man 
  
  ####Inputvariables for ML+man seeting  
  inputvarman2 = c(inputvar, "fy2_con_ana_n","fy2_con_ana_min","fy2_con_ana_max","fy2_con_ana_mean","fy2_con_ana_std" )
  
  clf =  h2o.randomForest(x= inputvarman2,y='F2eps_IBES_trim',training_frame=(training_all2),validation_frame = (validation_all2))
  yhatbig1= predict(clf,newdata = newdataset[,inputvarman2],type = 'response')
  
  
  
  MLman2[m]= mean(abs(yhatbig1-newdataset$F2eps_IBES)<abs(newdataset$ana_forecast_f2eps-newdataset$F2eps_IBES))
  
  
  
  
  
  
  ########predict analyst error  ML only 
  
  
  clf =  h2o.randomForest(x= inputvar,y='error_f2',training_frame=training_all2,validation_frame = validation_all2)
  
  yhatbig1= predict(clf,newdata = newdataset[,inputvar],type = 'response')+ newdataset$ana_forecast_f2eps
  
  
  
  
  
  
  error_MLonly2[m]= mean(abs(yhatbig1-newdataset$F2eps_IBES)<abs(newdataset$ana_forecast_f2eps-newdataset$F2eps_IBES))
  
  
  
  ########predict analyst error  ML+man only 
  clf = h2o.randomForest(x= inputvarman2,y='error_f2',training_frame=training_all1,validation_frame = validation_all1)
  
  yhatbig1= predict(clf,newdata = newdataset[,inputvarman2],type = 'response')+ newdataset$ana_forecast_f2eps
  error_MLman2[m]= mean(abs(yhatbig1-newdataset$F2eps_IBES)<abs(newdataset$ana_forecast_f2eps-newdataset$F2eps_IBES))
  h2o.removeAll()
}


h2o.shutdown(F)



dataplot=as.data.frame(cbind(end_dates,MLonly2,MLman2,error_MLonly2,error_MLman2))
dataplot = dataplot[-nrow(dataplot),]
colnames(dataplot) = c('end_dates','MLonly2','MLman2','error_MLonly2','error_MLman2')

dataplot2 =melt(dataplot, id.vars = c('end_dates'))


ggplot(data = dataplot2 ,aes(x=end_dates,y=value,group = variable,color= variable))+geom_line()+ggtitle('2 year ahead forecasting')



