from sklearn.linear_model import LinearRegression
import pandas as pd

# y = mx + b

df = pd.read_csv("Student_Marks.csv")
#print(df.head(3))

df.columns = ["sinif", "saat", "puan"]

y = df[["puan"]]
x = df[["sinif", "saat"]]

#df.info()

lm = LinearRegression()

model = lm.fit(x,y)

#print(model.predict([[3,2]])) # predict tahmin yapiyor

#print(df['Marks'].max())

#print(model.coef_) #model.coef_  kat sayi

#print(model.intercept_) #sabit

s1 = model.predict([[3,4.508]])[0][0]
print(s1)

print(model.score(x, y)) # ortalama sapma
