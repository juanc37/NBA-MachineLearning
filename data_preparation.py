#############
#  Author: Caleb Gelnar
#############

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.impute import SimpleImputer


def prepare_data(test_size, seed):
    data = pd.read_csv('nba_5_year.csv')

    X = data.drop(['TARGET_5Yrs', 'Name'], axis=1)
    Y = data.TARGET_5Yrs

    Simple_Imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    Simple_Imp.fit(X)
    X = Simple_Imp.transform(X)
    X = normalize(X)
    X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=test_size, random_state=seed)

    return X_Train, X_Test, Y_Train, Y_Test
