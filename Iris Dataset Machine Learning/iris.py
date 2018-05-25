import sklearn.datasets as datasets

import pandas as pd

from sklearn.cross_validation import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
import numpy as np

from sklearn.externals.six import StringIO

from IPython.display import Image

from sklearn.tree import export_graphviz

import pydotplus

import graphviz

def find_err(pred):	#function to find the accuracy basically
    diff_arr = np.equal(y_test, pred)
    correct_answers = np.sum(diff_arr)
    percent_diff = correct_answers / len(pred) * 100
    print("Iris: Percentage Match is: ", percent_diff)


iris = datasets.load_iris()	#Loads the dataset

df = pd.DataFrame(iris.data, columns = iris.feature_names)	#Create a dataframe of all independent variables

y = iris.target		#contains here is the dependent variable

print(df.head())	#Prints first 5 values in dataframe

x_train, x_test, y_train, y_test = train_test_split(df, y, test_size = 0.4)	#Splitting the dataset into test and training with 40% for testing 

dtree = DecisionTreeClassifier()	#Using a decision tree classifier model
dtree.fit(x_train, y_train)		#Getting the best fit by giving in the x_train and y_train

pred = dtree.predict(x_test)		#Predict the values of dependent variable by giving in x_test values

print(pred.sum())
print(y_test.sum())


    
find_err(pred)

cm_decil = pd.crosstab(y_test, pred, rownames=['True'], colnames=['Predicted'],margins = True)	#It's a confusion matrix that helps in checking which values are rightly mapped

print(cm_decil)

'''dot_data = StringIO()

export_graphviz(dtree, out_file = dot_data, filled = True, rounded = True, special_characters = True)

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())

Image(graph.create_png())'''


#print(df)
#print(y)

#print(len(x_train))
#print(len(x_test))

#print(x_test.head())
#print(x_train.head())
