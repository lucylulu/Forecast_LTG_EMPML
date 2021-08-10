
#####################
#This is part of the replication of "Empirical Asset Pricing via Machine Learning" (2018) and "Autoencoder Asset Pricing Models." (2019) 
#The aim of the file is data cleaning and seperation. 
#Data seperation: split data into y: dependent variables 
#x: all the possible independent variabls  
###### 
rm(list = ls()) 

library(data.table) 
library(dplyr) 
library(reshape2) 
library(fastDummies)
#Read data   

###If you have the dataset locally: 
####Method 1: read data from local. Set the path to the location of your data 

path ='/Users/xiazou/Desktop/Tinbergen_Study/block5/Forecasting_LTG/Replication/EAP_via_ML'

pathdata = paste0(path,'/GKX_20201231.csv')
data_all_ch = fread(pathdata)

pathmacro = paste0(path,'/PredictorData2020.csv')

data_all_macro = fread(pathmacro)

###Method 2: if you don't have the dataset locally: read it from dropbox directly 
# dropbox_link_ch = 'https://www.dropbox.com/s/ep97m74yn2znwn7/GKX_20201231.csv?dl=1'
# dropbox_link_macro ='https://www.dropbox.com/s/e18t43yc8eb3yj5/PredictorData2020.csv?dl=1'
# data_all_ch = fread(dropbox_link_ch) 
# data_all_macro = fread(dropbox_link_macro) 


##### 
dta_sub_ch = data_all_ch
rm(data_all_ch)

dta_sub_ch[,'DATE'] = as.Date(as.character(dta_sub_ch$DATE),'%Y%m%d')

#####choose data between the march 1957 to december 2016 

start_date = as.Date('19570301','%Y%m%d')
end_date = as.Date('20170101','%Y%m%d')
dta_sub_ch = subset(dta_sub_ch, DATE >start_date & DATE <end_date)
dta_sub_ch[,'DATE_ym'] = format(as.Date(dta_sub_ch$DATE),'%Y-%m')
####Choose macro data between the march 1957 to december 2016 
data_all_macro[,'DATE'] = data_all_macro[,'yyyymm']
data_all_macro[,'DATE']= as.Date(paste(as.character(data_all_macro$DATE),'01'),'%Y%m%d') 
data_all_macro_sub = subset(data_all_macro,DATE >= start_date & DATE <end_date)

data_all_macro_sub[,'DATE_ym'] = format(as.Date(data_all_macro_sub$DATE),'%Y-%m')
###Create term spread data 'tms' ,'dfy'

data_all_macro_sub[,'tms']= data_all_macro_sub$ltr - data_all_macro_sub$tbl
data_all_macro_sub[,'dfy']= data_all_macro_sub$BAA - data_all_macro_sub$AAA

###Choose variables that used in the paper 

macro_variable = c('D12','E12','b/m','ntis','tbl','tms','dfy','svar') 
macro_set_var = c('yyyymm','DATE','DATE_ym',macro_variable)
data_all_macro_sub1 = select(data_all_macro_sub,macro_set_var)



###############missing data for characteristics data  ***
##First step: delete dates that has variables with all na
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

na_ind_dtach = dta_sub_ch %>% group_by(DATE) %>% summarise_all(all_na_ind)



na_ind_dtach = as.data.frame(na_ind_dtach) 


######Replace dates with all nas with 0  

all_na = as.data.frame(colSums(na_ind_dtach[,-1]))

#variables with all na 
var_all_na = colnames(na_ind_dtach)[which(all_na$`colSums(na_ind_dtach[, -1])` != 0)+1 ]

for (var_na in var_all_na){ 
  id_na = which(colnames(dta_sub_ch) == var_na)
  date_na = na_ind_dtach[na_ind_dtach[[id_na]]==1,'DATE'] 
  dta_sub_ch[which(dta_sub_ch$DATE %in% date_na),][[id_na]] = 0
}






###Our start date is 1985-01-31

#Subset of non_all na obs 

#dta_sub_ch_1 = subset(dta_sub_ch, DATE %in% na_ind_dtach[rowSums(as.data.frame(na_ind_dtach[,-1]))==0,]$DATE)




impute_median <- function(x){
  ind_na = is.na(x) 
  x[ind_na] = median(x[!ind_na]) 
  return(as.numeric(x))
}

dta_sub_ch_2 = dta_sub_ch[,-102] %>% group_by(DATE)%>% mutate_all(impute_median)

dta_sub_ch_2[,'DATE_ym']= dta_sub_ch[,'DATE_ym']
rm(dta_sub_ch)




###Choose subset of data   would be comment out if we want to use the whole dataset 
index_all_sub = unique(dta_sub_ch_2$permno)  


time_all_sub = unique(dta_sub_ch_2$DATE) 

N_sub = length(index_all_sub)
T_sub = length(unique(dta_sub_ch_2$DATE))

#We have 32 years of obs, 18 years for training, 10 years for validation, 4 years for testing 



N_sub_1 = 500

index_sub_1 =index_all_sub[sample(1:N_sub,N_sub_1)]

dta_sub_ch_3 = subset(dta_sub_ch_2,permno %in% index_sub_1)

#######Merge characteristic data with macro data  
dta_sub_ch_3$DATE_ym = as.character(dta_sub_ch_3$DATE_ym)
data_all_macro_sub1$DATE_ym = as.character(data_all_macro_sub1$DATE_ym)
dta_sub_ch_31 = left_join(dta_sub_ch_3,data_all_macro_sub1,by='DATE_ym')
rm(dta_sub_ch_3)

Y_all = dta_sub_ch_31[,c('permno','DATE.x','RET','DATE_ym')]

char_name = colnames(dta_sub_ch_2)[-c(1,2,4,5,6,17,99,102)]

X_all_name = c('permno','DATE.x','DATE_ym',char_name)

X_all = dta_sub_ch_31[,char_name]

###
industry = dta_sub_ch_31[,c('permno','DATE.x','sic2')]
industry$sic2 = as.factor(as.integer(industry$sic2))
industry = dummy_cols(industry)


####Generate intersection terms 


Macro_inter = dta_sub_ch_31[,macro_variable]
X_all_inter = dta_sub_ch_31[,X_all_name]



for (mac_var in macro_variable){
  Macro_var = Macro_inter[,mac_var][[mac_var]]
  inter_X_M = X_all * Macro_var 
  X_all_inter = cbind(X_all_inter,inter_X_M)
}




#####Write data as csv
path_data = paste0(path,'/Data_Cleaned')
dir.create(path_data)


####Writing Y  
setwd(path_data) 
#Y 
fwrite(Y_all[,'RET'],'Y_cleaned.csv') 
#X
X_cleaned =  cbind(X_all_inter[,-c(1:3)],industry[,-c(1:3)])
  
fwrite(X_cleaned,'X_cleaned.csv')
#Index date 
fwrite(Y_all[,'DATE.x'],'Date_list.csv')

fwrite(Y_all[,'permno'],'permno_list.csv')






