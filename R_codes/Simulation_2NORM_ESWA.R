#############################################################
##########  SIMULATION STUDY: MULTIPLE EXPERIMENTS ##########
### EXPERIMENT 2 (NORMAL DISTRIBUTION WITH CORR MATRIX 2) ###
#############################################################
# Packages
library(MASS) # package needed to generate variables from a multivariate normal distribution
library(caret) # package needed for cross-validation
library(ineq) # package needed for the LZ computation
library(dplyr) # package needed to order the column of a dataframe
library(tidyr) # package needed to order the column of a dataframe
library(bootstrap) # package needed for implementing the Jackknife method
library(ggplot2) # package needed for plots
library(forcats)  # package needed for improve graphical representations
library(rpart) # package for implementing the regression tree
library(randomForest) # package for implementing random forest

# Generate a 4-dimensional multivariate normal with names (Y,X1,X2,X3,X4), and 100 observations, 
# with bilateral correlations
options(max.print=200000)

# Define correlation matrix (such that Y is lowly correlated with X1 (0.1), quite strongly correlated with X2 (0.5), 
# quite lowly correlated with X3 (0.3) and strongly correlated with X4 (0.8))

# Set the correlation matrix

S<-matrix(c(1.0, 0.1, 0.5, 0.2, 0.8,
            0.1, 1.0, 0.3, 0.6, 0.3,
            0.5, 0.3, 1.0, 0.05, 0.1,  
            0.2, 0.6, 0.05, 1.0, 0.5,
            0.8, 0.3, 0.1, 0.5, 1.0), ncol=5)

colnames(S)<-c("Y","X1","X2","X3","X4") # assign the names of the columns and the rows of the correlation matrix
rownames(S)<-colnames(S)

# Specify the mean vector
mu=c(150,20,72,5,200) 
names(mu)=colnames(S)

# number of observations in each dataset
n<-100

dfs<-list() # an empty list to hold the 1000 datasets

# Generate 1000 different datasets
s<-1000

for (i in 1:s)
{
  set.seed(i)
  dataframe<-as.data.frame(mvrnorm(n,mu=mu,Sigma=S)) # generate 10,000 observations 
  Y<-dataframe[,1]
  X1<-dataframe[,2]
  X2<-dataframe[,3]
  X3<-dataframe[,4]
  data<-data.frame(dataframe)
  colnames(data)[1]<-"Y"
  colnames(data)[2]<-"X1"
  colnames(data)[3]<-"X2"
  colnames(data)[4]<-"X3"
  colnames(data)[5]<-"X4"
  dfs[[i]]<-data
}

training.samples<-list()
train<-list()
test<-list()

## Split dataset into train and test (70% train and 30% test)
for (t in 1:s)
{
  set.seed(t)
  training.samples<-createDataPartition(Y,p=0.8,list=FALSE,times=1) 
  train[[t]]<-dfs[[t]][training.samples,] # training dataset
  test[[t]]<-dfs[[t]][-training.samples,] # test dataset
}

############################################################################################################################
# ACCURACY
RGA<-function(y,yhat){   
  ryhat<-rank(round(yhat,4),ties.method="min") # ranks of the predicted values
  support<-tapply(y,ryhat,mean) # replace the observed target variable value corresponding to the same predictive values with their mean 
  rord<-c(1:length(y))   
  for(jj in 1:length(y))
  {
    rord[jj]<-support[names(support)==ryhat[jj]] 
  }  
  ystar<-rord[order(yhat)] # re-order the observed target variable values with respect to the corresponding predicted values
  I<-1:length(y) 
  conc<-2*sum(I*ystar) # first term of the RGA numerator (concordance)
  dec<-2*sum(I*sort(y,decreasing=TRUE)) # second term of the RGA numerator and denominator (dual Lorenz)
  inc<-2*sum(I*sort(y)) # first term of the RGA denominator (Lorenz)
  RGA<-(conc-dec)/(inc-dec)
  RGA
}

# Set the objects for the inclusion of the predictions
pred_lin_reg<-list()
pred_reg_tree<-list()
pred_rf_mod<-list()

# Compute the RGA in each selected sample
RGA_lin_reg<-list() 
RGA_reg_tree<-list() 
RGA_rf_mod<-list()

for (r in 1:s)
{
  lin_reg<-lm(Y~.,data=train[[r]])
  pred_lin_reg[[r]]<-lin_reg%>%predict(test[[r]])
  RGA_lin_reg[[r]]<-round(RGA(test[[r]]$Y,pred_lin_reg[[r]]),3)
  
  reg_tree<-rpart(Y~.,data=train[[r]],method="anova")
  pred_reg_tree[[r]]<-reg_tree%>%predict(test[[r]])
  RGA_reg_tree[[r]]<-round(RGA(test[[r]]$Y,pred_reg_tree[[r]]),3)
  
  rf_mod<-randomForest(Y~.,data=train[[r]]) 
  pred_rf_mod[[r]]<-rf_mod%>%predict(test[[r]]) 
  RGA_rf_mod[[r]]<-round(RGA(test[[r]]$Y,pred_rf_mod[[r]]),3) # RGA for random forest
  
}

# Specify the RGA values in each sample
RGA_lin_reg<-unlist(RGA_lin_reg)
RGA_reg_tree<-unlist(as.vector(RGA_reg_tree))
RGA_rf_mod<-unlist(as.vector(RGA_rf_mod))

############################################################################################################################
# ROBUSTNESS (SUSTAINABILITY)
RGR<-function(yhat,yhat_pert){   
  ryhat_pert<-rank(round(yhat_pert,4),ties.method="min") # ranks of the predicted values
  support<-tapply(yhat,ryhat_pert,mean) # replace the observed target variable value corresponding to the same predictive values with their mean 
  rord<-c(1:length(yhat))   
  for(jj in 1:length(yhat))
  {
    rord[jj]<-support[names(support)==ryhat_pert[jj]] 
  }  
  yhatstar<-rord[order(yhat_pert)] # re-order the observed target variable values with respect to the corresponding predicted values
  I<-1:length(yhatstar) 
  conc<-2*sum(I*yhatstar) # first term of the RGR numerator (concordance)
  dec<-2*sum(I*sort(yhat,decreasing=TRUE)) # second term of the RGR numerator and denominator (dual Lorenz)
  inc<-2*sum(I*sort(yhat)) # first term of the RGR denominator (Lorenz)
  RGR<-(conc-dec)/(inc-dec)
  RGR
}

# Set the objects for the inclusion of the predictions from original data
pred_lin_reg<-list()
pred_reg_tree<-list()
pred_rf_mod<-list()

# Set the objects for the inclusion of the predictions from perturbed data
pred_lin_reg_pert<-list()
pred_reg_tree_pert<-list()
pred_rf_mod_pert<-list()

# Compute the RGR in each selected sample
RGR_lin_reg<-list() 
RGR_reg_tree<-list() 
RGR_rf_mod<-list()

for (r in 1:s)
{
  lin_reg<-lm(Y~.,data=train[[r]])
  pred_lin_reg[[r]]<-lin_reg%>%predict(test[[r]])
  
  reg_tree<-rpart(Y~.,data=train[[r]],method="anova")
  pred_reg_tree[[r]]<-reg_tree%>%predict(test[[r]])
  
  rf_mod<-randomForest(Y~.,data=train[[r]]) 
  pred_rf_mod[[r]]<-rf_mod%>%predict(test[[r]]) 
  
  # Perturbing data
  # Variable X1
  train[[r]]$X1<-replace(train[[r]]$X1,train[[r]]$X1>quantile(train[[r]]$X1,0.85),runif(n=15,min=15,max=22))
  train[[r]]$X1<-replace(train[[r]]$X1,train[[r]]$X1<quantile(train[[r]]$X1,0.15),runif(n=15,min=-6,max=-4))
  
  # Variable X2
  train[[r]]$X2<-replace(train[[r]]$X2,train[[r]]$X2>quantile(train[[r]]$X2,0.85),runif(n=15,min=15,max=22))
  train[[r]]$X2<-replace(train[[r]]$X2,train[[r]]$X2<quantile(train[[r]]$X2,0.15),runif(n=15,min=-6,max=-4))
  
  # Variable X3
  train[[r]]$X3<-replace(train[[r]]$X3,train[[r]]$X3>quantile(train[[r]]$X3,0.85),runif(n=15,min=15,max=22))
  train[[r]]$X3<-replace(train[[r]]$X3,train[[r]]$X3<quantile(train[[r]]$X3,0.15),runif(n=15,min=-6,max=-4))
  
  # Variable X4
  train[[r]]$X4<-replace(train[[r]]$X4,train[[r]]$X4>quantile(train[[r]]$X4,0.85),runif(n=15,min=15,max=22))
  train[[r]]$X4<-replace(train[[r]]$X4,train[[r]]$X4<quantile(train[[r]]$X4,0.15),runif(n=15,min=-6,max=-4))
  
  lin_reg_pert<-lm(Y~.,data=train[[r]])
  pred_lin_reg_pert[[r]]<-lin_reg_pert%>%predict(test[[r]])
  RGR_lin_reg[[r]]<-round(RGR(pred_lin_reg[[r]],pred_lin_reg_pert[[r]]),3) 
  
  reg_tree_pert<-rpart(Y~.,data=train[[r]],method="anova")
  pred_reg_tree_pert[[r]]<-reg_tree_pert%>%predict(test[[r]])
  RGR_reg_tree[[r]]<-round(RGR(pred_reg_tree[[r]],pred_reg_tree_pert[[r]]),3) 
  
  rf_mod_pert<-randomForest(Y~.,data=train[[r]]) 
  pred_rf_mod_pert[[r]]<-rf_mod_pert%>%predict(test[[r]]) 
  RGR_rf_mod[[r]]<-round(RGR(pred_rf_mod[[r]],pred_rf_mod_pert[[r]]),3) 
  
}

# Specify the RGR values in each sample
RGR_lin_reg<-unlist(RGR_lin_reg)
RGR_reg_tree<-unlist(as.vector(RGR_reg_tree))
RGR_rf_mod<-unlist(as.vector(RGR_rf_mod))

############################################################################################################################
# EXPLAINABILITY
RGEstar<-function(yhat,yhat_xk){   
  ryhat_xk<-rank(round(yhat_xk,4),ties.method="min") # ranks of the predicted values
  support<-tapply(yhat,ryhat_xk,mean) # replace the observed target variable value corresponding to the same predictive values with their mean 
  rord<-c(1:length(yhat))   
  for(jj in 1:length(yhat))
  {
    rord[jj]<-support[names(support)==ryhat_xk[jj]] 
  }  
  yhatstar<-rord[order(yhat_xk)] # re-order the observed target variable values with respect to the corresponding predicted values
  I<-1:length(yhatstar) 
  conc<-2*sum(I*yhatstar) # first term of the RGEstar numerator (concordance)
  dec<-2*sum(I*sort(yhat,decreasing=TRUE)) # second term of the RGEstar numerator and denominator (dual Lorenz)
  inc<-2*sum(I*sort(yhat)) # first term of the RGEstar denominator (Lorenz)
  RGEstar<-(conc-dec)/(inc-dec)
  RGEstar
}

# Set the objects for the inclusion of the predictions from the full model
pred_lin_reg<-list()
pred_reg_tree<-list()
pred_rf_mod<-list()

# Set the objects for the inclusion of the predictions from the reduced model without X1
pred_lin_reg_X1<-list()
pred_reg_tree_X1<-list()
pred_rf_mod_X1<-list()

# Set the objects for the inclusion of the predictions from the reduced model without X2
pred_lin_reg_X2<-list()
pred_reg_tree_X2<-list()
pred_rf_mod_X2<-list()

# Set the objects for the inclusion of the predictions from the reduced model without X3
pred_lin_reg_X3<-list()
pred_reg_tree_X3<-list()
pred_rf_mod_X3<-list()

# Set the objects for the inclusion of the predictions from the reduced model without X4
pred_lin_reg_X4<-list()
pred_reg_tree_X4<-list()
pred_rf_mod_X4<-list()

# Compute the RGE star in each selected sample
RGEstar_lin_reg_X1<-list() 
RGEstar_reg_tree_X1<-list() 
RGEstar_rf_mod_X1<-list()

RGEstar_lin_reg_X2<-list() 
RGEstar_reg_tree_X2<-list() 
RGEstar_rf_mod_X2<-list()

RGEstar_lin_reg_X3<-list() 
RGEstar_reg_tree_X3<-list() 
RGEstar_rf_mod_X3<-list()

RGEstar_lin_reg_X4<-list() 
RGEstar_reg_tree_X4<-list() 
RGEstar_rf_mod_X4<-list()

for (r in 1:s)
{
  lin_reg<-lm(Y~.,data=train[[r]])
  pred_lin_reg[[r]]<-lin_reg%>%predict(test[[r]])
  
  reg_tree<-rpart(Y~.,data=train[[r]],method="anova")
  pred_reg_tree[[r]]<-reg_tree%>%predict(test[[r]])
  
  rf_mod<-randomForest(Y~.,data=train[[r]]) 
  pred_rf_mod[[r]]<-rf_mod%>%predict(test[[r]]) 
  
  # Deletion of X1
  lin_reg_X1<-lm(Y~.-X1,data=train[[r]])
  pred_lin_reg_X1[[r]]<-lin_reg_X1%>%predict(test[[r]])
  
  reg_tree_X1<-rpart(Y~.-X1,data=train[[r]],method="anova")
  pred_reg_tree_X1[[r]]<-reg_tree_X1%>%predict(test[[r]])
  
  rf_mod_X1<-randomForest(Y~.-X1,data=train[[r]]) 
  pred_rf_mod_X1[[r]]<-rf_mod_X1%>%predict(test[[r]]) 
  
  # Variable X2
  lin_reg_X2<-lm(Y~.-X2,data=train[[r]])
  pred_lin_reg_X2[[r]]<-lin_reg_X2%>%predict(test[[r]])
  
  reg_tree_X2<-rpart(Y~.-X2,data=train[[r]],method="anova")
  pred_reg_tree_X2[[r]]<-reg_tree_X2%>%predict(test[[r]])
  
  rf_mod_X2<-randomForest(Y~.-X2,data=train[[r]]) 
  pred_rf_mod_X2[[r]]<-rf_mod_X2%>%predict(test[[r]]) 
  
  # Variable X3
  lin_reg_X3<-lm(Y~.-X3,data=train[[r]])
  pred_lin_reg_X3[[r]]<-lin_reg_X3%>%predict(test[[r]])
  
  reg_tree_X3<-rpart(Y~.-X3,data=train[[r]],method="anova")
  pred_reg_tree_X3[[r]]<-reg_tree_X3%>%predict(test[[r]])
  
  rf_mod_X3<-randomForest(Y~.-X3,data=train[[r]]) 
  pred_rf_mod_X3[[r]]<-rf_mod_X3%>%predict(test[[r]]) 
  
  # Variable X4
  lin_reg_X4<-lm(Y~.-X4,data=train[[r]])
  pred_lin_reg_X4[[r]]<-lin_reg_X4%>%predict(test[[r]])
  
  reg_tree_X4<-rpart(Y~.-X4,data=train[[r]],method="anova")
  pred_reg_tree_X4[[r]]<-reg_tree_X4%>%predict(test[[r]])
  
  rf_mod_X4<-randomForest(Y~.-X4,data=train[[r]]) 
  pred_rf_mod_X4[[r]]<-rf_mod_X4%>%predict(test[[r]]) 
  
  RGEstar_lin_reg_X1[[r]]<-round(RGEstar(pred_lin_reg[[r]],pred_lin_reg_X1[[r]]),3) 
  RGEstar_reg_tree_X1[[r]]<-round(RGEstar(pred_reg_tree[[r]],pred_reg_tree_X1[[r]]),3) 
  RGEstar_rf_mod_X1[[r]]<-round(RGEstar(pred_rf_mod[[r]],pred_rf_mod_X1[[r]]),3) 
  
  RGEstar_lin_reg_X2[[r]]<-round(RGEstar(pred_lin_reg[[r]],pred_lin_reg_X2[[r]]),3) 
  RGEstar_reg_tree_X2[[r]]<-round(RGEstar(pred_reg_tree[[r]],pred_reg_tree_X2[[r]]),3) 
  RGEstar_rf_mod_X2[[r]]<-round(RGEstar(pred_rf_mod[[r]],pred_rf_mod_X2[[r]]),3) 
  
  RGEstar_lin_reg_X3[[r]]<-round(RGEstar(pred_lin_reg[[r]],pred_lin_reg_X3[[r]]),3) 
  RGEstar_reg_tree_X3[[r]]<-round(RGEstar(pred_reg_tree[[r]],pred_reg_tree_X3[[r]]),3) 
  RGEstar_rf_mod_X3[[r]]<-round(RGEstar(pred_rf_mod[[r]],pred_rf_mod_X3[[r]]),3) 
  
  RGEstar_lin_reg_X4[[r]]<-round(RGEstar(pred_lin_reg[[r]],pred_lin_reg_X4[[r]]),3) 
  RGEstar_reg_tree_X4[[r]]<-round(RGEstar(pred_reg_tree[[r]],pred_reg_tree_X4[[r]]),3) 
  RGEstar_rf_mod_X4[[r]]<-round(RGEstar(pred_rf_mod[[r]],pred_rf_mod_X4[[r]]),3) 
  
}

# Specify the RGEstar values in each sample
# Variable X1
RGEstar_lin_reg_X1<-unlist(RGEstar_lin_reg_X1)
RGE_lin_reg_X1<-1-RGEstar_lin_reg_X1

RGEstar_reg_tree_X1<-unlist(as.vector(RGEstar_reg_tree_X1))
RGE_reg_tree_X1<-1-RGEstar_reg_tree_X1

RGEstar_rf_mod_X1<-unlist(as.vector(RGEstar_rf_mod_X1))
RGE_rf_mod_X1<-1-RGEstar_rf_mod_X1

# Variable X2
RGEstar_lin_reg_X2<-unlist(RGEstar_lin_reg_X2)
RGE_lin_reg_X2<-1-RGEstar_lin_reg_X2

RGEstar_reg_tree_X2<-unlist(as.vector(RGEstar_reg_tree_X2))
RGE_reg_tree_X2<-1-RGEstar_reg_tree_X2

RGEstar_rf_mod_X2<-unlist(as.vector(RGEstar_rf_mod_X2))
RGE_rf_mod_X2<-1-RGEstar_rf_mod_X2

# Variable X3
RGEstar_lin_reg_X3<-unlist(RGEstar_lin_reg_X3)
RGE_lin_reg_X3<-1-RGEstar_lin_reg_X3

RGEstar_reg_tree_X3<-unlist(as.vector(RGEstar_reg_tree_X3))
RGE_reg_tree_X3<-1-RGEstar_reg_tree_X3

RGEstar_rf_mod_X3<-unlist(as.vector(RGEstar_rf_mod_X3))
RGE_rf_mod_X3<-1-RGEstar_rf_mod_X3

# Variable X4
RGEstar_lin_reg_X4<-unlist(RGEstar_lin_reg_X4)
RGE_lin_reg_X4<-1-RGEstar_lin_reg_X4

RGEstar_reg_tree_X4<-unlist(as.vector(RGEstar_reg_tree_X4))
RGE_reg_tree_X4<-1-RGEstar_reg_tree_X4

RGEstar_rf_mod_X4<-unlist(as.vector(RGEstar_rf_mod_X4))
RGE_rf_mod_X4<-1-RGEstar_rf_mod_X4

############################################################################################################################
# FAIRNESS
RGF<-function(yhat,yhat_xg){   
  ryhat_xg<-rank(round(yhat_xg,4),ties.method="min") # ranks of the predicted values
  support<-tapply(yhat,ryhat_xg,mean) # replace the full model predicted values corresponding to the same reduced model predictive values with their mean
  rord<-c(1:length(yhat))   
  for(jj in 1:length(yhat))
  {
    rord[jj]<-support[names(support)==ryhat_xg[jj]] 
  }  
  yhatstar<-rord[order(yhat_xg)] # re-order the full model predicted values with respect to the corresponding reduced model predicted values
  I<-1:length(yhatstar) 
  conc<-2*sum(I*yhatstar) # first term of the RGF numerator (concordance)
  dec<-2*sum(I*sort(yhat,decreasing=TRUE)) # second term of the RGEstar numerator and denominator (dual Lorenz)
  inc<-2*sum(I*sort(yhat)) # first term of the RGF denominator (Lorenz)
  RGF<-(conc-dec)/(inc-dec)
  RGF
}

# Set the objects for the inclusion of the predictions from the full model
pred_lin_reg<-list()
pred_reg_tree<-list()
pred_rf_mod<-list()

# Set the objects for the inclusion of the predictions from the reduced model without 
# the binarised X1 variable (dummy)
pred_lin_reg_X1bin<-list()
pred_reg_tree_X1bin<-list()
pred_rf_mod_X1bin<-list()

# Compute the RGF in each selected sample
RGF_lin_reg_X1bin<-list() 
RGF_reg_tree_X1bin<-list() 
RGF_rf_mod_X1bin<-list()

for (r in 1:s)
{
  # Binarise the X1 variable in the train set
  train[[r]]$X1[train[[r]]$X1<mean(train[[r]]$X1)]<-0
  train[[r]]$X1[train[[r]]$X1>=mean(train[[r]]$X1)]<-1
  
  lin_reg<-lm(Y~.,data=train[[r]])
  pred_lin_reg[[r]]<-lin_reg%>%predict(test[[r]])
  
  reg_tree<-rpart(Y~.,data=train[[r]],method="anova")
  pred_reg_tree[[r]]<-reg_tree%>%predict(test[[r]])
  
  rf_mod<-randomForest(Y~.,data=train[[r]]) 
  pred_rf_mod[[r]]<-rf_mod%>%predict(test[[r]]) 
  
  # Deletion of X1
  lin_reg_X1bin<-lm(Y~.-X1,data=train[[r]])
  pred_lin_reg_X1bin[[r]]<-lin_reg_X1bin%>%predict(test[[r]])
  
  reg_tree_X1bin<-rpart(Y~.-X1,data=train[[r]],method="anova")
  pred_reg_tree_X1bin[[r]]<-reg_tree_X1bin%>%predict(test[[r]])
  
  rf_mod_X1bin<-randomForest(Y~.-X1,data=train[[r]]) 
  pred_rf_mod_X1bin[[r]]<-rf_mod_X1bin%>%predict(test[[r]]) 
  
  RGF_lin_reg_X1bin[[r]]<-round(RGF(pred_lin_reg[[r]],pred_lin_reg_X1bin[[r]]),3) 
  RGF_reg_tree_X1bin[[r]]<-round(RGF(pred_reg_tree[[r]],pred_reg_tree_X1bin[[r]]),3) 
  RGF_rf_mod_X1bin[[r]]<-round(RGF(pred_rf_mod[[r]],pred_rf_mod_X1bin[[r]]),3) 
  
}

RGF_lin_reg_X1bin<-unlist(RGF_lin_reg_X1bin)
RGF_reg_tree_X1bin<-unlist(as.vector(RGF_reg_tree_X1bin))
RGF_rf_mod_X1bin<-unlist(as.vector(RGF_rf_mod_X1bin))

#####################################################################
############################ BOXPLOTS ###############################
#####################################################################
# RGA, RGR and RGF
data<-data.frame(RGA_lin_reg,RGA_reg_tree,RGA_rf_mod,
                 RGR_lin_reg,RGR_reg_tree,RGR_rf_mod,
                 RGF_lin_reg_X1bin,RGF_reg_tree_X1bin,RGF_rf_mod_X1bin)

boxplot(data,las=2,col=c("lightskyblue1","aquamarine1","khaki1",
                         "lightskyblue1","aquamarine1","khaki1",
                         "lightskyblue1","aquamarine1","khaki1"),
        
        at=c(1,2,3, 5,6,7, 9,10,11),par(mar=c(5,4,4,2)+0.1),main="RGA, RGR, RGF: Normal Distribution - Correlation matrix S2", 
        ylim=c(0,1),names=c("RGA","RGA","RGA",
                            "RGR","RGR","RGR",
                            "RGF","RGF","RGF"))

abline(v=4,col="black",lty=7,lwd=1)
abline(v=8,col="black",lty=7,lwd=1)


add_legend <- function(...) {
  opar <- par(fig=c(0, 1, 0, 1), oma=c(0.5, 0.5, 0.5, 0.5), 
              mar=c(0, 0, 0, 0), new=TRUE)
  on.exit(par(opar))
  plot(0, 0, type='n', bty='n', xaxt='n', yaxt='n')
  legend(...)
}

add_legend("bottomleft",legend=c("Linear Regression","Regression tree","Random Forest"),fill=c("lightskyblue1","aquamarine1","khaki1"),horiz=FALSE,cex=0.55)

######################################################################
############################ BOXPLOTS  ###############################
######################################################################
# RGE
data<-data.frame(RGE_lin_reg_X1,RGE_reg_tree_X1,RGE_rf_mod_X1,
                 RGE_lin_reg_X2,RGE_reg_tree_X2,RGE_rf_mod_X2,
                 RGE_lin_reg_X3,RGE_reg_tree_X3,RGE_rf_mod_X3,
                 RGE_lin_reg_X4,RGE_reg_tree_X4,RGE_rf_mod_X4)

boxplot(data,las=2,col=c("royalblue","aquamarine1","khaki1",
                         "royalblue","aquamarine1","khaki1",
                         "royalblue","aquamarine1","khaki1",
                         "royalblue","aquamarine1","khaki1"),
        
        at=c(1,2,3, 5,6,7, 9,10,11, 13,14,15),par(mar=c(5,4,4,2)+0.1),main="RGE: Normal Distribution - Correlation matrix S2", 
        ylim=c(0,1),names=c("RGE X1","RGE X1","RGE X1",
                            "RGE X2","RGE X2","RGE X2",
                            "RGE X3","RGE X3","RGE X3",
                            "RGE X4","RGE X4","RGE X4"))

abline(v=4,col="black",lty=7,lwd=1)
abline(v=8,col="black",lty=7,lwd=1)
abline(v=12,col="black",lty=7,lwd=1)


add_legend <- function(...) {
  opar <- par(fig=c(0, 1, 0, 1), oma=c(0.5, 0.5, 0.5, 0.5), 
              mar=c(0, 0, 0, 0), new=TRUE)
  on.exit(par(opar))
  plot(0, 0, type='n', bty='n', xaxt='n', yaxt='n')
  legend(...)
}

add_legend("bottomleft",legend=c("Linear Regression","Regression tree","Random Forest"),fill=c("royalblue","aquamarine1","khaki1"),horiz=FALSE,cex=0.55)

##############################
### DESCRIPTIVE STATISTICS ###
##############################
# RGA
# Mean value
mean_RGA_lin_reg<-mean(RGA_lin_reg)
round(mean_RGA_lin_reg,4)
mean_RGA_reg_tree<-mean(RGA_reg_tree)
round(mean_RGA_reg_tree,4)
mean_RGA_rf_mod<-mean(RGA_rf_mod)
round(mean_RGA_rf_mod,4)

# Standard deviation
sd_RGA_lin_reg<-sd(RGA_lin_reg)
round(sd_RGA_lin_reg,4)
sd_RGA_reg_tree<-sd(RGA_reg_tree)
round(sd_RGA_reg_tree,4)
sd_RGA_rf_mod<-sd(RGA_rf_mod)
round(sd_RGA_rf_mod,4)

# RGR
# Mean value
mean_RGR_lin_reg<-mean(RGR_lin_reg)
round(mean_RGR_lin_reg,4)
mean_RGR_reg_tree<-mean(RGR_reg_tree)
round(mean_RGR_reg_tree,4)
mean_RGR_rf_mod<-mean(RGR_rf_mod)
round(mean_RGR_rf_mod,4)

# Standard deviation
sd_RGR_lin_reg<-sd(RGR_lin_reg)
round(sd_RGR_lin_reg,4)
sd_RGR_reg_tree<-sd(RGR_reg_tree)
round(sd_RGR_reg_tree,4)
sd_RGR_rf_mod<-sd(RGR_rf_mod)
round(sd_RGR_rf_mod,4)

# RGF
# Mean value
mean_RGF_lin_reg_X1bin<-mean(RGF_lin_reg_X1bin)
round(mean_RGF_lin_reg_X1bin,4)
mean_RGF_reg_tree_X1bin<-mean(RGF_reg_tree_X1bin)
round(mean_RGF_reg_tree_X1bin,4)
mean_RGF_rf_mod_X1bin<-mean(RGF_rf_mod_X1bin)
round(mean_RGF_rf_mod_X1bin,4)

# Standard deviation
sd_RGF_lin_reg_X1bin<-sd(RGF_lin_reg_X1bin)
round(sd_RGF_lin_reg_X1bin,4)
sd_RGF_reg_tree_X1bin<-sd(RGF_reg_tree_X1bin)
round(sd_RGF_reg_tree_X1bin,4)
sd_RGF_rf_mod_X1bin<-sd(RGF_rf_mod_X1bin)
round(sd_RGF_rf_mod_X1bin,4)

# RGE
# Variable X1
# Mean value
mean_RGE_lin_reg_X1<-mean(RGE_lin_reg_X1)
round(mean_RGE_lin_reg_X1,4)
mean_RGE_reg_tree_X1<-mean(RGE_reg_tree_X1)
round(mean_RGE_reg_tree_X1,4)
mean_RGE_rf_mod_X1<-mean(RGE_rf_mod_X1)
round(mean_RGE_rf_mod_X1,4)

# Standard deviation
sd_RGE_lin_reg_X1<-sd(RGE_lin_reg_X1)
round(sd_RGE_lin_reg_X1,4)
sd_RGE_reg_tree_X1<-sd(RGE_reg_tree_X1)
round(sd_RGE_reg_tree_X1,4)
sd_RGE_rf_mod_X1<-sd(RGE_rf_mod_X1)
round(sd_RGE_rf_mod_X1,4)

# Variable X2
# Mean value
mean_RGE_lin_reg_X2<-mean(RGE_lin_reg_X2)
round(mean_RGE_lin_reg_X2,4)
mean_RGE_reg_tree_X2<-mean(RGE_reg_tree_X2)
round(mean_RGE_reg_tree_X2,4)
mean_RGE_rf_mod_X2<-mean(RGE_rf_mod_X2)
round(mean_RGE_rf_mod_X2,4)

# Standard deviation
sd_RGE_lin_reg_X2<-sd(RGE_lin_reg_X2)
round(sd_RGE_lin_reg_X2,4)
sd_RGE_reg_tree_X2<-sd(RGE_reg_tree_X2)
round(sd_RGE_reg_tree_X2,4)
sd_RGE_rf_mod_X2<-sd(RGE_rf_mod_X2)
round(sd_RGE_rf_mod_X2,4)

# Variable X3
# Mean value
mean_RGE_lin_reg_X3<-mean(RGE_lin_reg_X3)
round(mean_RGE_lin_reg_X3,4)
mean_RGE_reg_tree_X3<-mean(RGE_reg_tree_X3)
round(mean_RGE_reg_tree_X3,4)
mean_RGE_rf_mod_X3<-mean(RGE_rf_mod_X3)
round(mean_RGE_rf_mod_X3,4)

# Standard deviation
sd_RGE_lin_reg_X3<-sd(RGE_lin_reg_X3)
round(sd_RGE_lin_reg_X3,4)
sd_RGE_reg_tree_X3<-sd(RGE_reg_tree_X3)
round(sd_RGE_reg_tree_X3,4)
sd_RGE_rf_mod_X3<-sd(RGE_rf_mod_X3)
round(sd_RGE_rf_mod_X3,4)

# Variable X4
# Mean value
mean_RGE_lin_reg_X4<-mean(RGE_lin_reg_X4)
round(mean_RGE_lin_reg_X4,4)
mean_RGE_reg_tree_X4<-mean(RGE_reg_tree_X4)
round(mean_RGE_reg_tree_X4,4)
mean_RGE_rf_mod_X4<-mean(RGE_rf_mod_X4)
round(mean_RGE_rf_mod_X4,4)

# Standard deviation
sd_RGE_lin_reg_X4<-sd(RGE_lin_reg_X4)
round(sd_RGE_lin_reg_X4,4)
sd_RGE_reg_tree_X4<-sd(RGE_reg_tree_X4)
round(sd_RGE_reg_tree_X4,4)
sd_RGE_rf_mod_X4<-sd(RGE_rf_mod_X4)
round(sd_RGE_rf_mod_X4,4)


