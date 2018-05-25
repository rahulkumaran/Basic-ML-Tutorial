import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
pd.options.mode.chained_assignment = None
from sklearn.externals import joblib


data = pd.read_csv("titanic_test.csv")

median_age = data['age'].median()
data['age'].fillna(median_age, inplace = True)

data_input = data[['pclass','age','sex']]

data_input['pclass'].replace('3rd', 3, inplace = True)
data_input['pclass'].replace('2nd', 2, inplace = True)
data_input['pclass'].replace('1st', 1, inplace = True)

data_input['sex'] = np.where(data_input['sex'] == 'female', 0, 1)

rf = joblib.load("titanic_survival_predictor_model")

pred = rf.predict(data_input)

print(pred)

def find_err(pred):
    titanic_data = np.loadtxt("titanic_results.txt", dtype="int32")
    diff_arr = np.equal(titanic_data, pred)
    correct_answers = np.sum(diff_arr)
    percent_diff = correct_answers / len(pred) * 100
    print("Titanic: Percentage Match is: ", percent_diff)
    
find_err(pred)

