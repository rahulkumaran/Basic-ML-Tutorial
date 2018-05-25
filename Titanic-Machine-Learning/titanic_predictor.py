import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
pd.options.mode.chained_assignment = None

from sklearn.externals import joblib

data = pd.read_csv("titanic_train.csv")

median_age = data['age'].median()
data['age'].fillna(median_age, inplace = True)

data_input = data[['pclass','age','sex']]
data_input.head()
expected_output = data[['survived']]
data_input['pclass'].replace('3rd', 3, inplace = True)
data_input['pclass'].replace('2nd', 2, inplace = True)
data_input['pclass'].replace('1st', 1, inplace = True)

data_input['sex'] = np.where(data_input['sex'] == 'female', 0, 1)

input_train, input_test, expected_op_train, expected_op_test = train_test_split(data_input, expected_output, test_size=0.33, random_state = 1000)


rf = RandomForestClassifier(n_estimators=100)

rf.fit(input_train, expected_op_train)

accuracy = rf.score(input_test, expected_op_test)
print("accuracy is {}%".format(accuracy*100))

joblib.dump(rf, "titanic_survival_predictor_model", compress = 9)

