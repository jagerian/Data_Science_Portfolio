## Intoduction:

library(dplyr)
library(lubridate)
library(stringr)
library(ggplot2)
library(dummies)
library(caret)
library(doParallel)

library(e1071)
library(class)
library(MASS)
library(rpart)
library(rpart.plot) ## install.packages("rpart.plot")
library(randomForest)
library(gbm)        ## install.packages("gbm")
library(C50)        ## install.packages("C50")

cidata <- read.csv("Data/carInsurance_train.csv", header=T) 

summary(cidata)
str(cidata)
ggplot(data = cidata) + geom_density(aes(x= cidata$Age), fill = "grey50")


####  변수
#### > Id
#### > Age
#### > Job
#### > Marital (결혼 상태)
#### > Education (교육 수준)
#### > Default (신용 파산)
#### > Balance (연봉)
#### > HHInsurance (가계 보험 여부)
#### > CarLoan (자동차 대출 여부)
#### > Communication (contact 방법, e.g. ‘celluar’, ‘ telephone’, ‘NA’)
#### > LastContactDay (최종 contact day)
#### > LastContactMonth (최종 contact month)
#### > NoOfContacts (현재 진행중인 캠페인 동안 contact한 횟수)
#### > DaysPassed (지난 캠페인 이후 지난 날 수)
#### > PrevAttempts (현재 진행중인 캠페인 이전에 contact한 횟수)
#### > Outcome (지난 캠페인의 outcome, e.g. ‘failure’, ‘other’, ‘success’, ’NA’)
#### > CallStart (최종 통화 시작 시간)
#### > CallEnd (최종 통화 종료 시간)
#### > CarInsurance (자동차 보험 가입 여부)




## string으로 된 시간데이터 정수형으로 바꾸기
starttime <- str_split(string = cidata$CallStart, pattern=":")
starttime <- data.frame(Reduce(rbind, starttime)) 
starttime$X1 <- as.numeric(as.character(starttime$X1))
starttime$X2 <- as.numeric(as.character(starttime$X2))
starttime$X3 <- as.numeric(as.character(starttime$X3))
startfactor <- c("starthour", "startminute", "startsecond")
names(starttime) <- startfactor
row.names(starttime) <- NULL
starttime


endtime <- str_split(string = cidata$CallEnd, pattern=":")
endtime <- data.frame(Reduce(rbind, endtime)) 
endtime$X1 <- as.numeric(as.character(endtime$X1))
endtime$X2 <- as.numeric(as.character(endtime$X2))
endtime$X3 <- as.numeric(as.character(endtime$X3))
endfactor <- c("endhour", "endminute", "endsecond")
names(endtime) <- endfactor
row.names(endtime) <- NULL
endtime

## Start와 End를 이용, 통화시간 구하기
dif <- endtime - starttime
dif <- (dif$endhour * 3600) + (dif$endminute * 60) + (dif$endsecond)
summary(dif)

cidata <- cbind(cidata, starttime, endtime)
cidata$dif <- dif
cidata <- cidata[,-c(17,18)]


## 나이 데이터 범주화, 독일의 법적기준에 따라 분류함
summary(cidata$Age)
ggplot(cidata, aes(y=cidata$Age, x=1)) + geom_violin()

cidata<- transform(cidata,
                   early_working_age = as.factor(ifelse(cidata$Age<=24, 1, 0)),
                   prime_working_age = as.factor(ifelse(cidata$Age>24 & cidata$Age<=54 , 1, 0)),
                   mature_working_age = as.factor(ifelse(cidata$Age>54 & cidata$Age<=64, 1, 0)),
                   elderly = as.factor(ifelse(cidata$Age>64, 1, 0)))


## https://www.indexmundi.com/germany/age_structure.html

cidata <- cidata[,-2]


## Missing Value 처리하기

sum(is.na(cidata$Job)) # 19개는 그냥 버림 귀찮음
cidata <- cidata[!c(is.na(cidata$Job)),] # 4000 -> 3981 obs

summary(cidata$Marital)


# Education Missing value
# NA -> idk
sum(is.na(cidata$Education))

cidata$Education_ <- addNA(cidata$Education)
levels(cidata$Education_) <- c(levels(cidata$Education), 'idk')

cidata$Educationf
cidata$Education <- NULL


# DaysPassed Missing value
# 1년 이상(>=365)인 경우 365로 초기화
# -1 인 신규고객 경우 역시 365
summary(cidata$DaysPassed)

ggplot(cidata, aes(y=cidata$DaysPassed, x=1)) + geom_violin()
ggplot(data = cidata) + geom_density(aes(x= cidata$DaysPassed), fill = "grey50")

cidata$DaysPassed[c(cidata$DaysPassed > 365)] <- 365
cidata$DaysPassed[c(cidata$DaysPassed == -1)] <- 365

summary(cidata$DaysPassed)
ggplot(data = cidata) + geom_density(aes(x= cidata$DaysPassed), fill = "grey50")

summary(cidata$Default)


# 이진 변수들 팩터화
cidata$Default <- as.factor(cidata$Default)
cidata$CarLoan <- as.factor(cidata$CarLoan)
cidata$CarInsurance <- as.factor(cidata$CarInsurance)
cidata$HHInsurance <- as.factor(cidata$HHInsurance)

# 이해 불가능한 두 변수 제거
cidata$LastContactDay <- NULL
cidata$LastContactMonth <- NULL

# Communication Missing value
# NA -> Missing
cidata$Communication_ <- addNA(cidata$Communication)
levels(cidata$Communication_) <- c(levels(cidata$Communication), 'Missing')

cidata$Communication <- NULL


# 의미 없는 데이터 제거
cidata$startminute <- NULL
cidata$startsecond <- NULL
cidata$endminute <- NULL
cidata$endsecond <- NULL

# 시간 팩터화
cidata$starthour <- as.factor(cidata$starthour)
cidata$endhour <- as.factor(cidata$endhour)

# Outocme Missing value
# NA -> Missing
cidata$Outcome_ <- addNA(cidata$Outcome)
levels(cidata$Outcome_) <- c(levels(cidata$Outcome), 'Missing')

cidata$Outcome <- NULL

## Balance 데이터 처리
# minusBalance 데이터 추가: Balance가 0 및 음수인 경우 1, 아닌경우 0
# Balance <= 1인경우 1로 초기화 후 Log변환하여 값을 모음

summary(cidata$Balance)

ggplot(data = cidata) + geom_density(aes(x= cidata$Balance), fill = "grey50")

cidata<- transform(cidata,
                   minusbalance = as.factor(ifelse(cidata$Balance<=1, 1, 0)))
                
cidata$logBalance <- cidata$Balance   
cidata$logBalance[c(cidata$Balance <= 1)] <- 1
summary(cidata$logBalance)

cidata$logBalance <- log10(cidata$logBalance)
ggplot(data = cidata) + geom_density(aes(x= cidata$logBalance), fill = "grey50")
cidata$Balance <- NULL

summary(cidata)
summary(cidata$starthour)
summary(cidata$endhour)
## 전처리 끝

## 팩터 -> 이진화
startdummy <- predict(dummyVars(~starthour, data = cidata), newdata = cidata)
startdummy <- as.data.frame(startdummy)

enddummy <- predict(dummyVars(~endhour, data = cidata), newdata = cidata)
enddummy <- as.data.frame(enddummy)

jobdummy <- predict(dummyVars(~Job, data = cidata), newdata = cidata)
jobdummy <- as.data.frame(jobdummy)

maritaldummy <- predict(dummyVars(~Marital, data = cidata), newdata = cidata)
maritaldummy <- as.data.frame(maritaldummy)

edudummy <- predict(dummyVars(~Education_, data = cidata), newdata = cidata)
edudummy <- as.data.frame(edudummy)

OCdummy <- predict(dummyVars(~Outcome_, data = cidata), newdata = cidata)
OCdummy <- as.data.frame(OCdummy)

CCdummy <- predict(dummyVars(~Communication_, data = cidata), newdata = cidata)
CCdummy <- as.data.frame(CCdummy)

cidata <- cbind(cidata,startdummy,enddummy,jobdummy,maritaldummy,edudummy,OCdummy,CCdummy)

cidata$Id <- NULL
cidata$Job <- NULL
cidata$Marital <- NULL
cidata$starthour <- NULL
cidata$endhour <- NULL
cidata$Education_ <- NULL
cidata$Communication_ <- NULL
cidata$Outcome_ <- NULL

summary(cidata)
## 진짜로 끝

cluster <- makeCluster(8)
registerDoParallel(cluster)
foreach::getDoParWorkers()

## 데이터 분할

cidata_success <- cidata[cidata$CarInsurance ==0,]
cidata_fail <- cidata[cidata$CarInsurance ==1,]

set.seed(0105)
sample_success <- sample(1:nrow(cidata_success), nrow(cidata_success)*0.75)
set.seed(0105)
sample_fail <- sample(1:nrow(cidata_fail), nrow(cidata_fail)*0.75)

cidata_train <- rbind(cidata_fail[sample_fail,], cidata_success[sample_success,])  # 2386 obs
cidata_test  <- rbind(cidata_fail[-sample_fail,], cidata_success[-sample_success,])#  996 obs

## 나이브 베이지언 ##

cidata_naive <- naiveBayes(CarInsurance~., data = cidata_train)
cidata_naive

pred_cidata_naive_tr <- predict(cidata_naive, newdata = cidata_train)
cidata_naive_tr_CM <- table(actual = cidata_train$CarInsurance, predicted = pred_cidata_naive_tr)
cidata_naive_tr_CM

naive_tr_acc <- (cidata_naive_tr_CM[1,1] + cidata_naive_tr_CM[2,2])/length(cidata_train$CarInsurance)

# 나이브베이지언 훈련집합 : 72.6 %

pred_cidata_naive_te <- predict(cidata_naive, newdata = cidata_test)
cidata_naive_te_CM <- table(actual = cidata_test$CarInsurance, predicted = pred_cidata_naive_te)
cidata_naive_te_CM

naive_te_acc <- (cidata_naive_te_CM[1,1] + cidata_naive_te_CM[2,2])/length(cidata_test$CarInsurance)

# 나이브베이지언 평가집합 : 76.1 %

result_naive <- as.data.frame(cbind(naive_tr_acc, naive_te_acc))
colnames(result_naive) <- c("tr_acc", "te_acc")
rownames(result_naive) <- c("Naïve Bayes")

## KNN ##

set.seed(0105)
tuning_knn <- tune.knn(x=cidata_train[,-7], y=cidata_train$CarInsurance, k=seq(3,19,by=2))
tuning_knn

set.seed(0105)
trControl <- trainControl(method  = "cv",
                          number  = 10)

set.seed(0105)
cidata_knn <- train(x=cidata_train[,-7], y=cidata_train[,7],
                 method     = "knn",
                 tuneGrid   = expand.grid(k = 7:13),
                 trControl  = trControl,
                 metric     = "Accuracy",
                 preProcess = c("center", "scale")
                 )

KNN_tr_acc <- max(cidata_knn$results$Accuracy)

# KNN 훈련집합 70.4% : k=11

pred_cidata_KNN_te <- predict(cidata_knn, newdata = cidata_test)
cidata_KNN_te_CM <- table(actual = cidata_test$CarInsurance, predicted = pred_cidata_naive_te)
cidata_KNN_te_CM

KNN_te_acc <- (cidata_KNN_te_CM[1,1] + cidata_KNN_te_CM[2,2])/length(cidata_test$CarInsurance)

# KNN 평가집합 76.1%

result_KNN <- as.data.frame(cbind(KNN_tr_acc, KNN_te_acc))
colnames(result_KNN) <- c("tr_acc", "te_acc")
rownames(result_KNN) <- c("K(11)-NN")


## LDA ##

cidata_LDA <- lda(CarInsurance~., data=cidata_train, cv=TRUE)

pred_cidata_LDA_tr <- predict(cidata_LDA, newdata = cidata_train)
cidata_LDA_tr_CM <- table(actual = cidata_train$CarInsurance, predicted = pred_cidata_LDA_tr$class)
cidata_LDA_tr_CM

LDA_tr_acc <- (cidata_LDA_tr_CM[1,1] + cidata_LDA_tr_CM[2,2])/length(cidata_train$CarInsurance)

# LDA 훈련집합 80.6%

pred_cidata_LDA_te <- predict(cidata_LDA, newdata = cidata_test)
cidata_LDA_te_CM <- table(actual = cidata_test$CarInsurance, predicted = pred_cidata_LDA_te$class)
cidata_LDA_te_CM

LDA_te_acc <- (cidata_LDA_te_CM[1,1] + cidata_LDA_te_CM[2,2])/length(cidata_test$CarInsurance)

# LDA 평가집합 81.7%

result_LDA <- as.data.frame(cbind(LDA_tr_acc, LDA_te_acc))
colnames(result_LDA) <- c("tr_acc", "te_acc")
rownames(result_LDA) <- c("LDA")

## 로지스틱 분류 ##

cidata_logit <- glm(CarInsurance~., data=cidata_train, family="binomial")
summary(cidata_logit)

pred_cidata_logit_tr <- predict(cidata_logit, newdata = cidata_train, type="response")
pred_cidata_logit_tr <- ifelse(pred_cidata_logit_tr>0.5, 1, 0) ## Threshold = 0.5
cidata_logit_tr_CM <- table(actual = cidata_train$CarInsurance, predict=pred_cidata_logit_tr)
cidata_logit_tr_CM

Logistic_tr_acc <- (cidata_logit_tr_CM[1,1] + cidata_logit_tr_CM[2,2])/length(cidata_train$CarInsurance)

# 로지스틱 분류 훈련집합 82.0%

pred_cidata_logit_te <- predict(cidata_logit, newdata = cidata_test, type="response")
pred_cidata_logit_te <- ifelse(pred_cidata_logit_te>0.5, 1, 0) ## Threshold = 0.5
cidata_logit_te_CM <- table(actual = cidata_test$CarInsurance, predict=pred_cidata_logit_te)
cidata_logit_te_CM

Logistic_te_acc <- (cidata_logit_te_CM[1,1] + cidata_logit_te_CM[2,2])/length(cidata_test$CarInsurance)

# 로지스틱 분류 평가집합 82.1%

result_Logit <- as.data.frame(cbind(Logistic_tr_acc, Logistic_te_acc))
colnames(result_Logit) <- c("tr_acc", "te_acc")
rownames(result_Logit) <- c("Logistic")

## Rpart 단일트리 ##

cidata_rpart <- rpart(CarInsurance~.,data = cidata_train, control=list(minsplit=15, minbucket=5))
cidata_rpart

pred_cidata_rpart_tr<- predict(cidata_rpart, type="class")
cidata_rpart_tr_CM <- table(actual = cidata_train$CarInsurance, predict=pred_cidata_rpart_tr)
cidata_rpart_tr_CM

rpart_tr_acc <- (cidata_rpart_tr_CM[1,1] + cidata_rpart_tr_CM[2,2])/length(cidata_train$CarInsurance)
rpart.plot(cidata_rpart)

# 단일 트리 rpart 훈련집합 79.3%

pred_cidata_rpart_te <- predict(cidata_rpart, newdata = cidata_test, type="class")
cidata_rpart_te_CM <- table(actual = cidata_test$CarInsurance, predicted = pred_cidata_rpart_te)
cidata_rpart_te_CM
rpart_te_acc <- (cidata_rpart_te_CM[1,1] + cidata_rpart_te_CM[2,2])/length(cidata_test$CarInsurance)

# 단일 트리 rpart 평가집합 80.6%

result_rpart <- as.data.frame(cbind(rpart_tr_acc, rpart_te_acc))
colnames(result_rpart) <- c("tr_acc", "te_acc")
rownames(result_rpart) <- c("rpart")


## C5.0 tree

cidata_C5.0_fit <- C5.0(CarInsurance~., data=cidata_train, control = C5.0Control(minCases=10))
summary(cidata_C5.0_fit)

pred_cidata_C5.0_tr<- predict(cidata_C5.0_fit, newdata = cidata_train, type="class")
cidata_C5.0_tr_CM <- table(actual = cidata_train$CarInsurance, predict=pred_cidata_C5.0_tr)
cidata_C5.0_tr_CM
C50_tr_acc <- (cidata_C5.0_tr_CM[1,1] + cidata_C5.0_tr_CM[2,2])/length(cidata_train$CarInsurance)

# C5.0 트리 훈련집합 82.1%

pred_cidata_C5.0_te<- predict(cidata_C5.0_fit, newdata = cidata_test, type="class")
cidata_C5.0_te_CM <- table(actual = cidata_test$CarInsurance, predict=pred_cidata_C5.0_te)
cidata_C5.0_te_CM
C50_te_acc <- (cidata_C5.0_te_CM[1,1] + cidata_C5.0_te_CM[2,2])/length(cidata_test$CarInsurance)

# C5.0 트리 평가집합 81.4%

result_c50 <- as.data.frame(cbind(C50_tr_acc, C50_te_acc))
colnames(result_c50) <- c("tr_acc", "te_acc")
rownames(result_c50) <- c("C5.0")

## C5.0 + boosting

cidata_C5.0B_fit <- C5.0(CarInsurance~., data=cidata_train, control = C5.0Control(minCases=10), trials=50)
summary(cidata_C5.0B_fit)

pred_cidata_C5.0B_tr<- predict(cidata_C5.0B_fit, newdata = cidata_train, type="class")
cidata_C5.0B_tr_CM <- table(actual = cidata_train$CarInsurance, predict=pred_cidata_C5.0B_tr)
cidata_C5.0B_tr_CM
C50B_tr_acc <- (cidata_C5.0B_tr_CM[1,1] + cidata_C5.0B_tr_CM[2,2])/length(cidata_train$CarInsurance)

# C5.0 트리 + 부스팅 훈련집합 84.1%

pred_cidata_C5.0B_te<- predict(cidata_C5.0B_fit, newdata = cidata_test, type="class")
cidata_C5.0B_te_CM <- table(actual = cidata_test$CarInsurance, predict=pred_cidata_C5.0B_te)
cidata_C5.0B_te_CM
C50B_te_acc <-(cidata_C5.0B_te_CM[1,1] + cidata_C5.0B_te_CM[2,2])/length(cidata_test$CarInsurance)

# C5.0 트리 평가집합 82.9%

result_c50B <- as.data.frame(cbind(C50B_tr_acc, C50B_te_acc))
colnames(result_c50B) <- c("tr_acc", "te_acc")
rownames(result_c50B) <- c("C5.0B")


## RandomForest

set.seed(0105)
tuning_rf <- tune.randomForest(x=cidata_train[,-7], y=cidata_train$CarInsurance, ntree=seq(50,150,by=10), mtry=3:5)
tuning_rf$best.parameters

set.seed(0105)
cidata_rf_fit <- randomForest(x=cidata_train[,-7],
                              y=cidata_train$CarInsurance,
                              ntree=tuning_rf$best.parameters[,2],
                              mtry=tuning_rf$best.parameters[,1],
                              do.trace=30,
                              nodesize=10,
                              importance=T,
                              data=cidata_train)
cidata_rf_fit
importance(cidata_rf_fit)

pred_cidata_rf_tr_CM <- predict(cidata_rf_fit, newdata=cidata_train, type="class")
cidata_rf_tr_CM <- table(actual = cidata_train$CarInsurance, predict = pred_cidata_rf_tr_CM)
cidata_rf_tr_CM
RF_tr_acc <- (cidata_rf_tr_CM[1,1] + cidata_rf_tr_CM[2,2])/length(cidata_train$CarInsurance)

# RandomForest 훈련집합 91.8%

pred_cidata_rf_te_CM <- predict(cidata_rf_fit, newdata=cidata_test, type="class")
cidata_rf_te_CM <- table(actual = cidata_test$CarInsurance, predict = pred_cidata_rf_te_CM)
cidata_rf_te_CM
RF_te_acc <- (cidata_rf_te_CM[1,1] + cidata_rf_te_CM[2,2])/length(cidata_test$CarInsurance)

# RandomForest 평가집합 82.7%

result_RF <- as.data.frame(cbind(RF_tr_acc, RF_te_acc))
colnames(result_RF) <- c("tr_acc", "te_acc")
rownames(result_RF) <- c("RandomForest")

## 선형 SVM

set.seed(0105)
tuning_linearSVM <- tune.svm(CarInsurance~., cost=c(seq(0.1,1,by=0.1),2:5),
                             kernel="linear", data=cidata_train)

set.seed(0105)
cidata_SVM_fit <- svm(CarInsurance~., cost=tuning_linearSVM$best.parameters[,1],
                      kernel="linear", data=cidata_train)
summary(cidata_SVM_fit)

pred_cidata_linearSVM_tr <- predict(cidata_SVM_fit, newdata=cidata_train)
cidata_SVM_tr_CM <- table(actual = cidata_train$CarInsurance, predicted = pred_cidata_linearSVM_tr)
cidata_SVM_tr_CM

Linear_SVM_tr_acc <- (cidata_SVM_tr_CM[1,1] + cidata_SVM_tr_CM[2,2])/length(cidata_train$CarInsurance)

# 선형 SVM 흔련집합 82.3%

pred_cidata_linearSVM_te <- predict(cidata_SVM_fit, newdata=cidata_test)
cidata_SVM_te_CM <- table(actual = cidata_test$CarInsurance, predicted = pred_cidata_linearSVM_te)
cidata_SVM_te_CM

Linear_SVM_te_acc <- (cidata_SVM_te_CM[1,1] + cidata_SVM_te_CM[2,2])/length(cidata_test$CarInsurance)

# 선형 SVM 평가집합 82.5%

result_Linear_SVM <- as.data.frame(cbind(Linear_SVM_tr_acc, Linear_SVM_te_acc))
colnames(result_Linear_SVM) <- c("tr_acc", "te_acc")
rownames(result_Linear_SVM) <- c("Linear_SVM")

## RBF SVM

set.seed(0105)
tuning_RBFSVM <- tune.svm(CarInsurance~., cost=c(seq(0.1,1,by=0.1),2:5),
                          kernel="radial", gamma=10^(-4:2), data=cidata_train)

set.seed(0105)
cidata_RBFSVM_fit <- svm(CarInsurance~., cost=tuning_RBFSVM$best.parameters[,2],
                         degree=tuning_RBFSVM$best.parameters[,1], kernel="radial",
                        data=cidata_train)


summary(cidata_RBFSVM_fit)

pred_cidata_RBFSVM_tr <- predict(cidata_RBFSVM_fit, newdata=cidata_train)
cidata_RBFSVM_tr_CM <- table(actual = cidata_train$CarInsurance, predicted = pred_cidata_RBFSVM_tr)
cidata_RBFSVM_tr_CM

RBF_SVM_tr_acc <- (cidata_RBFSVM_tr_CM[1,1] + cidata_RBFSVM_tr_CM[2,2])/length(cidata_train$CarInsurance)

# RBF SVM 흔련집합 88.2%

pred_cidata_RBFSVM_te <- predict(cidata_SVM_fit, newdata=cidata_test)
cidata_RBFSVM_te_CM <- table(actual = cidata_test$CarInsurance, predicted = pred_cidata_RBFSVM_te)
cidata_RBFSVM_te_CM

RBF_SVM_te_acc <- (cidata_RBFSVM_te_CM[1,1] + cidata_RBFSVM_te_CM[2,2])/length(cidata_test$CarInsurance)

# RBF SVM 평가집합 82.5%

result_RBF_SVM <- as.data.frame(cbind(RBF_SVM_tr_acc, RBF_SVM_te_acc))
colnames(result_RBF_SVM) <- c("tr_acc", "te_acc")
rownames(result_RBF_SVM) <- c("RBF_SVM")

result_table <- rbind(result_naive,
                      result_KNN,
                      result_Logit,
                      result_LDA,
                      result_rpart,
                      result_c50,
                      result_c50B,
                      result_RF,
                      result_Linear_SVM,
                      result_RBF_SVM)

## 결과 정리

