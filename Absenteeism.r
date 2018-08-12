#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees','RRF')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

library(rpart)

#Reading the input data
absenteeism_data = read.csv("Absenteeism_at_work.csv")

#Converting necessary predictors to categorical

absenteeism_data$Reason.for.absence = as.factor(absenteeism_data$Reason.for.absence)
absenteeism_data$Month.of.absence = as.factor(absenteeism_data$Month.of.absence)
absenteeism_data$Day.of.the.week = as.factor(absenteeism_data$Day.of.the.week)
absenteeism_data$Seasons = as.factor(absenteeism_data$Seasons)
absenteeism_data$Disciplinary.failure = as.factor(absenteeism_data$Disciplinary.failure)
absenteeism_data$Education = as.factor(absenteeism_data$Education)
absenteeism_data$Son = as.factor(absenteeism_data$Son)
absenteeism_data$Social.drinker = as.factor(absenteeism_data$Social.drinker)
absenteeism_data$Social.smoker = as.factor(absenteeism_data$Social.smoker)
absenteeism_data$Pet = as.factor(absenteeism_data$Pet)

#Converting empty string to NA
absenteeism_data$Work.load.Average.day[absenteeism_data$Work.load.Average.day == ""] = NA

#Renaming workload average/day to workload
names(absenteeism_data)[10] = "workload"

#Removing the commas from the workload column so that we could use it as a number
absenteeism_data$workload = gsub(",", "", absenteeism_data$workload)

absenteeism_data$workload = as.integer(absenteeism_data$workload)



### MISSING VALUE ANALYSIS ###

missing_val = data.frame(apply(absenteeism_data,2,function(x){sum(is.na(x))}))

#Creating a dataframe to store the number of missing values for each attributes.
missing_val$Columns = row.names(missing_val)

#Rename variable to proper format
names(missing_val)[1] =  "Missing_percentage"

#Calculate percentage of missing values for each variable
missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(absenteeism_data)) * 100

#Arranging in descending order
missing_val = missing_val[order(-missing_val$Missing_percentage),]
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]

#Imputation process showed below. Please uncomment lines and run it, if you want to check

#setting the first value in Body.mass.index column as nan 
#Original value = 30

#absenteeism_data$Body.mass.index[1] = NA

#Imputing with mean. Value obtained = 26.67938

#absenteeism_data$Body.mass.index[is.na(absenteeism_data$Body.mass.index)] = mean(absenteeism_data$Body.mass.index, na.rm = T)

#Imputing with median. Value obtained = 25

#absenteeism_data$Body.mass.index[is.na(absenteeism_data$Body.mass.index)] = median(absenteeism_data$Body.mass.index, na.rm = T)

#KNN Imputation. Value obtained = 30
absenteeism_data = knnImputation(absenteeism_data, k = 3)



### OUTLIER ANALYSIS ###

#Saving the numeric columns first, as outlier analysis is performed on numerical values:

numeric_index = sapply(absenteeism_data,is.numeric)

numeric_data = absenteeism_data[,numeric_index]

cnames = colnames(numeric_data)

#Plot boxplot to visualize outliers:

for (i in 1:length(cnames))
   {
     assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "Absenteeism.time.in.hours"), data = subset(absenteeism_data))+ 
              stat_boxplot(geom = "errorbar", width = 0.5) +
              geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                           outlier.size=1, notch=FALSE) +
              theme(legend.position="bottom")+
              labs(y=cnames[i],x="Absenteeism.time.in.hours")+
              ggtitle(paste("Box plot of responded for",cnames[i])))
}

### Plotting plots together
 #gridExtra::grid.arrange(gn1,gn5,gn2,ncol=3)
 #gridExtra::grid.arrange(gn6,gn7,ncol=2)
 #gridExtra::grid.arrange(gn8,gn9,ncol=2)
 
 
  #loop to remove all outliers
  for(i in cnames){
    print(i)
    val = absenteeism_data[,i][absenteeism_data[,i] %in% boxplot.stats(absenteeism_data[,i])$out]
    absenteeism_data = absenteeism_data[which(!absenteeism_data[,i] %in% val),]
  }
 

 
 ### FEATURE SELECTION ###
 
## Correlation Plot 

 corrgram(absenteeism_data[,numeric_index], order = TRUE, lower.panel=panel.shade, upper.panel=panel.pie, text.panel=panel.txt, main = "Correlation Plot")

 
 ## Chi-squared Test of Independence
 # Finding p value from the Chi square test, to check the significance of the independent and dependent variable.
 factor_index = sapply(absenteeism_data,is.factor)
 factor_data = absenteeism_data[,factor_index]
 
 for (i in 1:11)
 {
   print(names(factor_data)[i])
   print(chisq.test(table(absenteeism_data$Absenteeism.time.in.hours,factor_data[,i])))
 }
 
 ## DIMENSION REDUCTION
 absenteeism_data = subset(absenteeism_data, 
                          select = -c(Service.time,Age,Weight,Education,Pet,ID,Day.of.the.week,Social.smoker))
 
 
 ##FEATURE SCALING
 
 #Normalisation
 #The predictors have different ranges. We hope to have its range in 0-1, so they have better scaling, and do not lead to biased predictions.
 cnames = c("Transportation.expense","Distance.from.Residence.to.Work","workload","Hit.target","Height","Body.mass.index")
 
 for(i in cnames){
   print(i)
   absenteeism_data[,i] = (absenteeism_data[,i] - min(absenteeism_data[,i]))/
     (max(absenteeism_data[,i] - min(absenteeism_data[,i])))
 }
 
 ### MODEL DEVELOPMENT ###
 absenteeism_data = subset(absenteeism_data, 
                           select = -c(Service.time,Age,Weight,Education,Pet,ID,Day.of.the.week,Social.smoker))
 rmExcept("absenteeism_data")
 
 #Divide data into train and test using stratified sampling method
 set.seed(1234)
 train.index = createDataPartition(absenteeism_data$Absenteeism.time.in.hours, p = .80, list = FALSE)
 train = absenteeism_data[ train.index,]
 test  = absenteeism_data[-train.index,]
 
 #Decision tree regression
 
 fit <- rpart(Absenteeism.time.in.hours ~ Reason.for.absence + Transportation.expense + Distance.from.Residence.to.Work , method = "anova", data = train)
 fit <- rpart(Absenteeism.time.in.hours ~ . , method = "anova", data = train)
 
 #Plotting the model 
 
 plot(fit, main = "Regression for Absenteeism Time in hours")
 text(fit, use.n = TRUE, all = TRUE)
 
 printcp(fit)
 par(mfrow=c(1,2)) 
 rsq.rpart(fit)
 
 #Predicting
 
 mean((test$Absenteeism.time.in.hours - predict(fit, test, method = "anova"))^2)
 
 ##Random forests####
 
 library(randomForest)
 
 fitt <- RRF(Absenteeism.time.in.hours ~ ., data = train, importance = TRUE)
 plot(fitt)
 
 mean((test$Absenteeism.time.in.hours - predict(fitt, test, method = "anova"))^2)
 
 varImpPlot(fitt)
 predict(fitt,test)
 
 summary(fitt)
 #######
 
 ##answering question 2
 
 for(month in 1:13){
   df = data.frame(Reason.for.absence = factor(), Month.of.absence = factor(), Seasons = factor(),
                   Transportation.expense = double(), Distance.from.Residence.to.Work = double(),
                   workload = double(), Hit.target = double(), Disciplinary.failure = factor(),
                   Son = factor(), Social.drinker = factor(), Height = double(), Body.mass.index = double(), Absenteeism.time.in.hours = double())
   for(i in 1:nrow(absenteeism_data)){
     if(absenteeism_data$Month.of.absence[i] == month){
       df = rbind(df,absenteeism_data[i,])
       
     }
   }
   RRF_Predictions = predict(fitt,df)
   print("Month: ")
   print(month)
   print("Losses to company: ")
   print(sum(RRF_Predictions))
 }
  
 
 
 
 
 
 
 
 
 
 
 
 
 
 