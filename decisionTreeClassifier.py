import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
import graphviz
from sklearn.model_selection import train_test_split

df = pd.read_csv('heart.csv')
print(df.head(3))

y = df['output']
x = df.drop('output', axis=1)

tree = DecisionTreeClassifier()
model = tree.fit(x,y)
skor = model.score(x,y)
print(skor)

#Train Test Split ile

x_train, x_test, y_train, y_test = train_test_split(x,y, random_state=16, train_size=0.70)

tree1 = DecisionTreeClassifier()
model1 = tree1.fit(x_train, y_train)
skor1 = model.score(x_test, y_test)
print(skor1)

model.predict([[31,1,2,130,240,0,0,150,0,2,0,0,2]])

dot = export_graphviz(model, feature_names=x.columns, filled=True)
gorsel = graphviz.Source(dot)
print(gorsel)