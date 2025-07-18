import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.model_selection import train_test_split

df = pd.read_csv("heart.csv")
df.head(3)

y = df['output']
x = df.drop(columns='output')

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=42, train_size=0.7)
dt = DecisionTreeClassifier()
model = dt.fit(x,y)
skor = model.score()
print(skor)

dt1 = RandomForestClassifier(n_estimators=200)
model1 = dt1.fit(x_train, y_train)
skor1 = model1.score(x_test, y_test)
print(skor1)

rf = xgb.XGBClassifier()
model2 = rf.fit(x_train, y_train)
skor2 = model.score(x_test, y_test)
print(skor2)

insan = df.sample().drop("output", axis=1).values
print(model.predict(insan))
