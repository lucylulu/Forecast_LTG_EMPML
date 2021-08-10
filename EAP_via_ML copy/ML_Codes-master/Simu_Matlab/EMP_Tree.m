%%%  This version: July 23
%%% This is to replicate "Empirical Asset Pricing via Machine Learning" (2018)

%%% All tree models %%% 

%%%%Path where the data located  
clear;
path='/Users/xiazou/Desktop/Tinbergen_Study/block5/Forecasting_LTG/Replication/EAP_via_ML'; % set your own folder path (the same with R folder path)    
dirdata=strcat(path,'/Data_Cleaned'); 


    

%%%%Make a dir for storing results  

title = strcat(dirdata,'/Tree') 
mkdir(title) 
titleB = sprintf('%s',title,'/B');
mkdir(titleB); 


%%%%%Read files  
pathY = strcat(dirdata,'/Y_cleaned.csv') 
pathX = strcat(dirdata,'/X_cleaned.csv') 
pathdate = strcat(dirdata,'/Date_list.csv') 

Y_all = csvread(pathY,1,0) 

%X = csvread(pathX,1,0)  
Date_Index = readtable(pathdate) 
Date_index1 =table2array(Date_Index)
Date_index = datetime(Date_index1,'InputFormat','yyyy-mm-dd')
[y_index,m_index,d_index] = ymd(Date_index)


DX = tabularTextDatastore(pathX)
%X = readall(DX)

X_all = [] ; 
while hasdata(DX)
    t= read(DX) ;
    X_all = vertcat(X_all,t); 
end 

X_all = table2array(X_all) 


%%%%%%%%%%%%%%Training, validation, testing set 

%%%Start date = 1985-01 , end date =  

%for training 
start_date = datetime(1985, 1, 1); 

end_date =  start_date + years(18) ;

ind_training = (start_date < Date_index) & (Date_index < end_date); 

ytrain = Y_all(ind_training,:) ;
xtrain = X_all(ind_training,:);  

%for validation 

start_date =  end_date 
end_date = start_date + years(10) 

ind_vali = (start_date < Date_index) & (Date_index < end_date); 

ytest = Y_all(ind_vali,:) ;
xtest= X_all(ind_vali,:);  

%for test  oos 

start_date =  end_date 
end_date = start_date + years(4)

ind_test = (start_date < Date_index) & (Date_index < end_date);

yoos = Y_all(ind_test,:) ;
xoos= X_all(ind_test,:);

%%% Monthly Demean %%%
ytrain_demean=ytrain-mean(ytrain);
ytest_demean=ytest-mean(ytest);
mtrain=mean(ytrain);
mtest=mean(ytest);

%%% Start to train %%%

r2_oos=zeros(3,1);  %%% OOS R2
r2_is=zeros(3,1);  %%% IS R2


nump = 940 ;
%lamv = 10:50:nump; %num of predictors to sample
lamv = 100;


ne=100;  %num of trees
%lamc = [2,4,8,16,32]; %maxinum of split
lamc = 8;
r=zeros(length(lamv),length(lamc),3);

for n1 = 1:length(lamv)
    nf=lamv(n1);
    for n2 = 1:length(lamc)
        nn=lamc(n2);
        clf=TreeBagger(ne,xtrain,ytrain,'Method','regression','NumPredictorsToSample',nf,'MaxNumSplits',nn);
        yhatbig1 = predict(clf,xtest);
        r(n1,n2,1)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
        yhatbig1 = predict(clf,xoos);
        r(n1,n2,2)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
        yhatbig1 = predict(clf,xtrain);
        r(n1,n2,3)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
    end
end

fw_2 = fw2(r(:,:,1));
r2_oos(1)=r(fw_2(1),fw_2(2),2);
r2_is(1)=r(fw_2(1),fw_2(2),3);
disp(strcat('RF R2 : ',num2str(r2_oos(1),3)));


%%% GBRT %%%

%lamv=-1:0.2:0;  
lamv = -0.6

r=zeros(length(lamv),50,3);

for n1 = 1: length(lamv)
    lr=10^lamv(n1);
    alpha=2;
    ne=50;
    t=templateTree('MaxNumSplits',2,'Surrogate','on');
    clf=fitensemble(xtrain,ytrain,'LSBoost',ne,t,'Type','regression','LearnRate',lr);
    
    % e=predict(clf,xtest);
    % e = error(clf,xtest,ytest);
    e=loss(clf,xtest,ytest,'mode','cumulative');
    for i = 1:length(e);
        r(n1,i,1) = e(i);
        % pred = e(i);
        % yhatbig1 = pred;
        % r(n1,i,1)=1-sum(power(yhatbig1-ytest,2))/sum(power(ytest-mtrain,2));
    end
    
    %e=error(clf,xoos,yoos);
    e=loss(clf,xoos,yoos,'mode','cumulative');
    for i = 1:length(e);
        r(n1,i,2) = e(i);
        % pred = e(i);
        % yhatbig1 = pred;
        % r(n1,i,2)=1-sum(power(yhatbig1-yoos,2))/sum(power(yoos-mtrain,2));
    end
    
    %e=error(clf,xtrain,ytrain);
    e=loss(clf,xtrain,ytrain,'mode','cumulative');
    for i = 1:length(e);
        r(n1,i,3) = e(i);
        % pred = e(i);
        % yhatbig1 = pred;
        % r(n1,i,2)=1-sum(power(yhatbig1-ytrain,2))/sum(power(ytrain-mtrain,2));
    end
    
end

fw_2 = fw2(-r(:,:,1));
err1=mean((ytrain-mtrain).^2);
err2=mean((yoos-mtrain).^2);
r2_oos(2)=1-r(fw_2(1),fw_2(2),2)/err2;
r2_is(2)=1-r(fw_2(1),fw_2(2),3)/err1;
disp(strcat('GBRT R2 : ',num2str(r2_oos(2),3)));

%disp(r2_oos)
pathr=sprintf('%s',title,'/roos');
%pathr=sprintf('%s_%d_%d',pathr);
pathb=sprintf('%s',pathr,'.csv');
csvwrite(pathr,r2_oos);

%disp(r2_is)
pathr=sprintf('%s',title,'/ris');
%pathr=sprintf('%s_%d_%d',pathr,mo,M);
pathb=sprintf('%s',pathr,'.csv');
csvwrite(pathr,r2_is);




            

