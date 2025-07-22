import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

df = pd.read_csv("plane.csv")
#print(df.head(3))
info = df.info()
#print(info)

df = df[["Rcmnd cruise Knots","Stall Knots dirty", "Fuel gal/lbs", "Eng out rate of climb", "Takeoff over 50ft", "Price"]]
#print(df.head(3))

y = df["Price"] # No normalization is performed for y.
x = df.drop("Price", axis=1)

# reduces the outlier effect
# model performance increases

ss = StandardScaler()
x2 = ss.fit_transform(x)

x2 = pd.DataFrame(x2)
print(x2.head(3))

x2[0].mean()
x2[0].std()

mm = MinMaxScaler()
x3 = mm.fit_transform(x)
x3 = pd.DataFrame(x3)

mm2 = MinMaxScaler(feature_range=(0,10)) # we can determine the range with feature_range
