import pandas as pd 
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

df = pd.DataFrame()
df['cumleler'] = ["ali bak", "ali ata bak", "bak ali bak", "ali güzel ata bak", "ışık ılık süt iç"]

cv = CountVectorizer(max_features=4)
a = cv.fit_transform(df['cumleler'])
x = a.toarray()
#print(x)

print(cv.get_feature_names_out())

# stop words like "ne, mesela, fakat, gibi, ... ", "this, as, am/is/are, to,  ..."