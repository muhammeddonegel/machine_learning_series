import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

df = pd.read_csv("insurance.csv")
#print(df.head(3))
df = pd.get_dummies(df, columns = ['sex', 'smoker', 'region'], drop_first = True)
y = df[["charges"]]
x = df.drop('charges', axis=1)

lm = LinearRegression()
model = lm.fit(x,y)

model.score(x,y)

#print(model.predict([[19, 26, 0, 1, 1, 0, 0, 1]]))

df_hata = pd.DataFrame()
df_hata['y'] = y
y_tahmin = model.predict(x)
df_hata['tahmin'] = y_tahmin
df_hata['error'] = y - y_tahmin


df_hata['squared_error'] = df_hata['error']**2 #mse yani karelerini almak
df_hata['abstract_error'] = np.abs(df_hata['error']) #mae yani mutlakdeger
df_hata['percent_error'] = np.abs((y-y_tahmin)/y) #mape

print(df_hata.mean())

# kisa yollari
mean_squared_error(y,y_tahmin) #mse 
mean_absolute_error(y, y_tahmin) #mae
mean_absolute_percentage_error(y, y_tahmin) #mape
