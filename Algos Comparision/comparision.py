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
data = pd.read_csv("Loan_Train.csv")
test_data = pd.read_csv("Loan_Test.csv")


#--------------------------------------------------- DATA CLEANING PROCESS -----------------------------------
data['Education'] = np.where(data['Education'] == 'Graduate', 1, 0)
data['Self_Employed'] = np.where(data['Self_Employed'] == 'Yes', 1, 0)
data['Loan_Status'] = np.where(data['Loan_Status'] == 'Y', 1, 0)
data['Married'] = np.where(data['Married'] == 'Yes', 1, 0)

data['Property_Area'].replace('Urban', 1, inplace  = True)
data['Property_Area'].replace('Semiurban', 2, inplace  = True)
data['Property_Area'].replace('Rural', 3, inplace  = True)
data['Dependents'].replace('3+', 3, inplace  = True)

data_inp = data[['Self_Employed','ApplicantIncome','CoapplicantIncome','LoanAmount','Loan_Amount_Term','Credit_History','Property_Area','Loan_Status']]
#print(data_inp.head())
#print(data_inp.isnull())	#Gender, self employed, Loan Amount, Loan Amount Term, Credit_History
#print(data_inp.corr())

inp = data_inp[['ApplicantIncome','CoapplicantIncome','Credit_History','LoanAmount','Loan_Amount_Term']]
expected_output = data_inp['Loan_Status']

test_inp = test_data[['ApplicantIncome','CoapplicantIncome','Credit_History','LoanAmount','Loan_Amount_Term']]

k_inp = data_inp[['ApplicantIncome','CoapplicantIncome','Credit_History','LoanAmount','Loan_Amount_Term','Loan_Status']]

inp['LoanAmount'].fillna(round(inp['LoanAmount'].mean()), inplace=True)
inp['Loan_Amount_Term'].fillna(360, inplace=True)
inp['Credit_History'].fillna(inp['Credit_History'].median(), inplace=True)
test_inp['LoanAmount'].fillna(round(test_inp['LoanAmount'].mean()), inplace=True)
test_inp['Loan_Amount_Term'].fillna(360, inplace=True)
test_inp['Credit_History'].fillna(test_inp['Credit_History'].median(), inplace=True)
k_inp['LoanAmount'].fillna(round(k_inp['LoanAmount'].mean()), inplace=True)
k_inp['Loan_Amount_Term'].fillna(360, inplace=True)
k_inp['Credit_History'].fillna(k_inp['Credit_History'].median(), inplace=True)
train_x, test_x, train_y, test_y = train_test_split(inp, expected_output, test_size = 0.3, random_state = 9999)


#---------------------------------------------------- LINEAR REGRESSION APPLIED -------------------------------
lr = LinearRegression()
model = lr.fit(train_x, train_y)
pred = model.predict(test_x)
conf = confusion_matrix(test_y, pred>=0.5)
#pred_test = model.predict(test_inp)
#pred_test = np.where(pred_test>=0.5, 'Y', 'N')
#print(pred_test)
print(conf)

#--------------------------------------------------  RANDOM FOREST APPLIED ------------------------------------

rf = RandomForestClassifier(n_estimators=40).fit(train_x, train_y)
pred = rf.predict(test_x)
conf_rf = confusion_matrix(test_y, pred>=0.5)
#print(conf_rf)


#------------------------------------------------- BAGGING CLASSIFIER APPLIED ---------------------------------
bagging = BaggingClassifier(KNeighborsClassifier(), max_samples=0.5, max_features=0.5)
model_bag = bagging.fit(train_x, train_y)
pred_bag = model_bag.predict(test_x)
conf_bag = confusion_matrix(test_y, pred_bag>=0.5)
#print(conf_bag)


#------------------------------------------------ ADA BOOSTING APPLIED ----------------------------------------
ada = AdaBoostClassifier(n_estimators=100)
model_ada = ada.fit(train_x, train_y)
pred_ada = model_ada.predict(test_x)
conf_ada = confusion_matrix(test_y, pred_ada>=0.5)
#print(conf_ada)


#----------------------------------------------- DECISION TREE CLASSIFIER -------------------------------------
dtree = DecisionTreeClassifier()
model_tree = dtree.fit(train_x, train_y)
pred_tree = dtree.predict(test_x)
conf_tree = confusion_matrix(test_y, pred_tree>=0.5)
#print(conf_tree)


#---------------------------------------------- KMEANS APPLIED ------------------------------------------------
kmeans = KMeans(n_clusters=2)
model = kmeans.fit(k_inp)
pred_kmeans = kmeans.predict(k_inp)
conf_k = confusion_matrix(k_inp[['Loan_Status']],pred_kmeans>=0.5)
#print(conf_k)


#---------------------------------------------- LOGISTIC REGRESSION APPLIED -----------------------------------
log = LogisticRegression()
model_log = log.fit(train_x, train_y)
pred_log = model_log.predict(test_x)
conf_log = confusion_matrix(test_y, pred_log)
print(conf_log)


#---------------------------------------------- NAIVE BAYES CLASSIFIER APPLIED --------------------------------
nb = GaussianNB()
model_nb = nb.fit(train_x,train_y)
pred_nb = nb.predict(test_x)
conf_nb = confusion_matrix(test_y, pred_nb)
print(conf_nb)





