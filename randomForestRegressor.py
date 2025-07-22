import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

df = pd.read_csv("insurance.csv")
print(df.head(3))

df = pd.get_dummies(df, columns=["sex", "smoker", "region"], drop_first=True)

y = df["charges"]
x = df.drop(columns=["charges"])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=22, train_size=0.7)

lr = LinearRegression()
model = lr.fit(x_train, y_train)
skor = model.score(x_test, y_test)
print(skor)

rf = RandomForestRegressor(n_estimators=200)
model1 = rf.fit(x_train, y_train)
skor2 = model1.score(x_test, y_test)
print(skor2)

