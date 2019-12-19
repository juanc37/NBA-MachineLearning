#############
#  Author: Caleb Gelnar and Juan Candelaria Claborne
#############

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sn
import matplotlib.pyplot as plt
data = pd.read_csv('596/nba_5_year.csv')
unique = data.apply(lambda x: len(x.unique()))

X = data.drop(['TARGET_5Yrs', 'Name'], axis=1)
Y = data.TARGET_5Yrs

Simple_Imp = SimpleImputer(missing_values=np.nan, strategy='mean')
Simple_Imp.fit(X)
X = Simple_Imp.transform(X)

corrMatrix = data.corr()
sn.heatmap(corrMatrix, annot=False)
plt.show()

print(data.head())
print()
print("######## UNIQUE DATA WITHIN EACH COLUMN #########")
print(unique)
print("##############################################")
print()
print("######## X SHAPE, Y SHAPE #########")
print(X.shape, Y.shape)
print("###################################")
print()
print("######## NUMBER OF \"0\" OUTPUTS #########")
print(len(data[data['TARGET_5Yrs'] == 0]))
print("##########################################")
print()
print("######## NUMBER OF \"1\" OUTPUTS #########")
print(len(data[data['TARGET_5Yrs'] == 1]))
print("##########################################")

