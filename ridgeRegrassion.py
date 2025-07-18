# Ridge regression  overfitting (Asiri ogrenme durumlari icin kullanirlir)
# Ridge regression sayesinde bias ve varyans arasindaki dengeyi saglayabilir.
# Ridge regression da katsayilar uzerinde regresyon yapilir
# ridge regression da katsayilar kuculur ama sifir olmaz. Features oz nitelik azalmaz.
# ridge regression cezalar karesi ile orantilidir

# y= a1 * x1 + a2 * x2 + ... + b + alfa * (katsayilar toplami)**2

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge

df = pd.read_csv("student_scores.csv")
print(df.head(3))

y = df["Scores"]
x = df[["Hours"]]

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8,8))
plt.scatter(x,y)
plt.show()

lr = LinearRegression()
model = lr.fit(x,y)
model.score(x,y)

alfalar = [1,10,20,100,200]
for a in alfalar:
    r = Ridge(alpha=a)
    modelr = r.fit(x,y)
    skor=modelr.score(x,y)
    print("Skor: ", skor)
    print("Katsayilar: ", modelr.coef_)
    