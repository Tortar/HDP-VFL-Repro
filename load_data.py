
import pandas as pd
import numpy as np
import statsmodels.api as sm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import normalize
 

"""
name should be either "adult", "breast", "credit"
"""

class DataLoader:
    def __init__(self, name):
        df = pd.read_csv("data/" + name + "/" + name + ".csv")
        df = df.dropna()
        df = pd.get_dummies(df, drop_first=True)
        X = df.iloc[:,:-1].to_numpy(dtype=np.float64)
        y = df.iloc[:,-1:].to_numpy(dtype=np.float64)
        X = sm.add_constant(X)
        X = normalize(X, axis=1, norm="l2")
        self.X_train, self.y_train = X, y

    def preprocess_adult_dataset(self, adult):

        # Replacing values in 'X2' column
        adult.loc[adult['X2'].isin(['Without-pay', 'Never-worked']), 'X2'] = 'Unemployed'
        adult.loc[adult['X2'].isin(['State-gov', 'Local-gov', 'Self-emp-inc', 'Self-emp-not-inc']), 'X2'] = 'Employed'

        # Replacing values in 'X4' column

        edu_levels = {"Preschool": 0, "1st-4th":0, "5th-6th": 1, "7th-8th": 1, "9th": 2, "10th": 2, "11th": 2,
                      "12th": 3, "HS-grad": 3, "Some-college": 3, "Assoc-voc": 4, "Assoc-acdm": 4, "Bachelors": 4,
                      "Prof-school": 5, "Masters": 5, "Doctorate": 6}
        
        for name_level in set(adult["X4"]): adult.loc[adult["X4"] == name_level, "X4"] = edu_levels[name_level]
        adult = adult.astype({'X4': 'int32'})

        # Replacing values in 'X6' column
        adult.loc[adult['X6'].isin(['Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent']), 'X6'] = 'Married'
        adult.loc[adult['X6'].isin(['Divorced', 'Separated', 'Widowed']), 'X6'] = 'Not-Married'

        # Replacing values in 'X14' column
        north_america = ["Canada", "Cuba", "Dominican-Republic", "El-Salvador", "Guatemala",
                         "Haiti", "Honduras", "Jamaica", "Mexico", "Nicaragua",
                         "Outlying-US(Guam-USVI-etc)", "Puerto-Rico", "Trinadad&Tobago",
                         "United-States"]
        asia = ["Cambodia", "China", "Hong", "India", "Iran", "Japan", "Laos",
                "Philippines", "Taiwan", "Thailand", "Vietnam"]
        south_america = ["Columbia", "Ecuador", "Peru"]
        europe = ["England", "France", "Germany", "Greece", "Holand-Netherlands",
                  "Hungary", "Ireland", "Italy", "Poland", "Portugal", "Scotland",
                  "Yugoslavia"]
        other = ["South"]

        adult.loc[adult['X14'].isin(north_america), 'X14'] = 'North America'
        adult.loc[adult['X14'].isin(asia), 'X14'] = 'Asia'
        adult.loc[adult['X14'].isin(south_america), 'X14'] = 'South America'
        adult.loc[adult['X14'].isin(europe), 'X14'] = 'Europe'
        adult.loc[adult['X14'].isin(other), 'X14'] = 'Other'

        adult = adult.drop(['X2', 'X6', 'X7', 'X8', 'X9', 'X10', 'X14'], axis=1)

        return adult

    def preprocess_breast_dataset(self, X, y):

        X = SelectKBest(f_classif, k=10).fit_transform(X, np.ravel(y))
        X += 0.08*np.random.default_rng(10).normal(size=X.shape)
        return X