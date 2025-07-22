# it is converte  categorical data to numeric data
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("tdf.csv")
#print(df.head(3))

le = LabelEncoder()
x = le.fit(df["Team"])

le.transform(df["Team"])
print(le.classes_)

le.inverse_transform(x)

le2 = LabelEncoder()
le2.fit_transform(df["Team"])