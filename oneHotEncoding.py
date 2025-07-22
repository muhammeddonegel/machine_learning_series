import pandas as pd 
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


df = pd.read_csv("churn.csv")
df = df.drop(columns=["RowNumber", "CustomerId", "Surname"])
#print(df.head(4))
ohe = OneHotEncoder()
xd = ohe.fit_transform(df[['Geography', 'Gender']]).toarray()
ohe.get_feature_names_out()
xd = pd.DataFrame(xd)
xd.columns = ohe.get_feature_names_out()
print(xd.head(4))
df = df.drop(columns=["Geography", "Gender"])
df[xd.columns] = xd

#df = pd.get_dummies(df, columns=["Geography", "Gender"], drop_first=True)

y = df["Exited"]
x = df.drop("Exited", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=61, train_size=0.7)

rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)
skor = model.score(x_test, y_test)
print(skor)

pre = model.predict([[712, 31, 9, 0, 5, 1, 1, 200000, False, False, True]])
print(pre)


#df = ["BJK", "BJK", "FB", "GS", "DB", "RM", "TS", "VS", "BAR", "AN", "XX", "XY", "OS"]

#from sklearn.preprocessing import OneHotEncoder

#ohe = OneHotEncoder()
#x = ohe.fit_transform(df[["takim"]])
#x.toarray()
#ohe.transform([["BJK"]]).toarray()