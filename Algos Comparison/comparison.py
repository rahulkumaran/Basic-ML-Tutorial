import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.naive_bayes import GaussianNB

#--------------------------------------------------- READING THE DATASET -------------------------------------
data = pd.read_csv("Loan_Train.csv")		#Reading the training dataset
test_data = pd.read_csv("Loan_Test.csv")	#Reading the testing dataset


#--------------------------------------------------- DATA CLEANING PROCESS -----------------------------------
data['Education'] = np.where(data['Education'] == 'Graduate', 1, 0)		#Replacing all words with numbers in education
data['Self_Employed'] = np.where(data['Self_Employed'] == 'Yes', 1, 0)		#Replacing all words with numbers in self_employed
data['Loan_Status'] = np.where(data['Loan_Status'] == 'Y', 1, 0)		#Replacing all words with numbers in loan_status
data['Married'] = np.where(data['Married'] == 'Yes', 1, 0)			#Replacing all words with numbers in married

data['Property_Area'].replace('Urban', 1, inplace  = True)			#Replacing all urban with 1s in property_area
data['Property_Area'].replace('Semiurban', 2, inplace  = True)			#Replacing all semiurban with 2s in property_area
data['Property_Area'].replace('Rural', 3, inplace  = True)			#Replacing all rural with 3s in property_area
data['Dependents'].replace('3+', 3, inplace  = True)				#Replacing all 3+'s with 3s in dependents

data_inp = data[['Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']]	#Choosing the necessary data
#print(data_inp.head())
#print(data_inp.isnull())	#Gender, self employed, Loan Amount, Loan Amount Term, Credit_History
#print(data_inp.corr())

inp = data_inp[['ApplicantIncome','CoapplicantIncome','Credit_History','LoanAmount','Loan_Amount_Term']]	#The input data to the model
expected_output = data_inp['Loan_Status']		#The output data for the model

test_inp = test_data[['ApplicantIncome','CoapplicantIncome','Credit_History','LoanAmount','Loan_Amount_Term']]	#The input data from test dataset

k_inp = data_inp[['ApplicantIncome','CoapplicantIncome','Credit_History','LoanAmount','Loan_Amount_Term','Loan_Status']]	#The input data for kmeans with the ouput var too

inp['LoanAmount'].fillna(round(inp['LoanAmount'].mean()), inplace=True)			#Filling all NaN's in Loan_amount with mean in input data
inp['Loan_Amount_Term'].fillna(360, inplace=True)					#Filling all NaN's with 360 in loan_amount_term
inp['Credit_History'].fillna(inp['Credit_History'].mean(), inplace=True)		#Filling all NaN's with median in credit_history
test_inp['LoanAmount'].fillna(round(test_inp['LoanAmount'].median()), inplace=True)
test_inp['Loan_Amount_Term'].fillna(360, inplace=True)
test_inp['Credit_History'].fillna(test_inp['Credit_History'].median(), inplace=True)
k_inp['LoanAmount'].fillna(round(k_inp['LoanAmount'].mean()), inplace=True)
k_inp['Loan_Amount_Term'].fillna(360, inplace=True)
k_inp['Credit_History'].fillna(k_inp['Credit_History'].median(), inplace=True)
train_x, test_x, train_y, test_y = train_test_split(inp, expected_output, test_size = 0.3, random_state = 9999)		#Splitting training set to train and validation data


#---------------------------------------------------- LINEAR REGRESSION APPLIED -------------------------------
lr = LinearRegression()			#Calling LinearRegression
model = lr.fit(train_x, train_y)	#Fitting and creating a model
pred = lr.predict(test_x)		#Predicting the answers for valdiation data
conf = confusion_matrix(test_y, pred>=0.5)	#Checking accuracy
pred_test = model.predict(test_inp)		#Predicting the values for test set
pred_test = np.where(pred_test>=0.5, 'Y', 'N')	#Changing all values greater than 0.5 to Y and less than that to N
frame = pd.DataFrame(pred_test)			#Changing the predictions to a dataframe

test_data = pd.concat([test_data,frame],axis=1)	#Merging the predictions to the test_data

#print(test_data.head())		#Checks head of test data
#print(pred_test)		#Prints the predicted values
#print(conf)			#Prints confusion matrix

test_data.to_csv("loan.csv")	#Creates a new data set with the name loan.csv that has the predictions

#--------------------------------------------------  RANDOM FOREST APPLIED ------------------------------------

rf = RandomForestClassifier(n_estimators=40).fit(train_x, train_y)	#Calling RandomForestClassifier and fitting to create model
pred = rf.predict(test_x)		#Predictions being made
conf_rf = confusion_matrix(test_y, pred>=0.5)	#Creating confusion matrix
#print(conf_rf)


#------------------------------------------------- BAGGING CLASSIFIER APPLIED ---------------------------------
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)	#Calling BaggingClassifier
model_bag = bagging.fit(train_x, train_y)		#Fitting to create model with training data
pred_bag = model_bag.predict(test_x)			#Predictions being made
conf_bag = confusion_matrix(test_y, pred_bag>=0.5)	#Confusion matrix created
#print(conf_bag)


#------------------------------------------------ ADA BOOSTING APPLIED ----------------------------------------
ada = AdaBoostClassifier(n_estimators=100)		#AdaBoostClassifier being called
model_ada = ada.fit(train_x, train_y)			#Model is being fit
pred_ada = model_ada.predict(test_x)			#predictions made
conf_ada = confusion_matrix(test_y, pred_ada>=0.5)	#Confusion matrix created
#print(conf_ada)


#----------------------------------------------- DECISION TREE CLASSIFIER -------------------------------------
dtree = DecisionTreeClassifier()	#Decision Tree Classifier being called
model_tree = dtree.fit(train_x, train_y)	#Model created with training data being fit
pred_tree = model_tree.predict(test_x)		#Preditctions made
conf_tree = confusion_matrix(test_y, pred_tree>=0.5)	#Confusion matrix created
#print(conf_tree)


#---------------------------------------------- KMEANS APPLIED ------------------------------------------------
kmeans = KMeans(n_clusters=2)	#KMeans being called 
model = kmeans.fit(k_inp)	#Model created with training data being fit
pred_kmeans = kmeans.predict(k_inp)		#Preditctions made
conf_k = confusion_matrix(k_inp[['Loan_Status']],pred_kmeans>=0.5)	#Confusion matrix created
#print(conf_k)


#---------------------------------------------- LOGISTIC REGRESSION APPLIED -----------------------------------
log = LogisticRegression()		#Logistic Regression being called
model_log = log.fit(train_x, train_y)		#Model created with training data being fit
pred_log = model_log.predict(test_x)		#Preditctions made
conf_log = confusion_matrix(test_y, pred_log)	#Confusion matrix created
#print(conf_log)


#---------------------------------------------- NAIVE BAYES CLASSIFIER APPLIED --------------------------------
nb = GaussianNB()		#NaiveBayes being called
model_nb = nb.fit(train_x,train_y)		#Model created with training data being fit
pred_nb = model_nb.predict(test_x)		#Preditctions made
conf_nb = confusion_matrix(test_y, pred_nb)	#Confusion matrix created
#print(conf_nb)


#---------------------------------------------- ACCURACY CALCULATIONS BEING MADE ------------------------------
print("\tACCURACIES OF VARIOUS ALGOS")
print("Linear Regression:",(conf[0,0]+conf[1,1])/conf.sum()*100,"%")
print("Random Forest Classifier:",(conf_rf[0,0]+conf_rf[1,1])/conf_rf.sum()*100,"%")
print("Bagging:",(conf_bag[0,0]+conf_bag[1,1])/conf_bag.sum()*100,"%")
print("Ada Boosting:",(conf_ada[0,0]+conf_ada[1,1])/conf_ada.sum()*100,"%")
print("Decision Tree Classifier:",(conf_tree[0,0]+conf_tree[1,1])/conf_tree.sum()*100,"%")
print("KMeans:",(conf_k[0,0]+conf_k[1,1])/conf_k.sum()*100,"%")
print("Logistic Regression:",(conf_log[0,0]+conf_log[1,1])/conf_log.sum()*100,"%")
print("Naive Bayes:",(conf_nb[0,0]+conf_nb[1,1])/conf_nb.sum()*100,"%")
