import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv("customer_booking.csv", encoding = "latin-1")
#print(df.head(3))

df = pd.get_dummies(df, columns=["sales_channel", "trip_type", "flight_day", "route", "booking_origin"], drop_first=True)

y = df["wants_extra_baggage"]
x = df.drop("wants_extra_baggage", axis=1)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42, train_size=0.85)

rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)
skor = model.score(x_test, y_test)
#print(skor)

crossVal = cross_val_score(model, x, y, cv = 4)
print(crossVal)
