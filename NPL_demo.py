import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv('banka.csv')
df = df[['metin', 'kategori']]

stopwords = ['fakat','lakin','ancak','acaba', 'ama', 'aslında', 'az', 'bazı', 'belki', 'biri', 'birkaç', 'birşey', 'biz', 'bu', 'çok', 'çünkü', 'da', 'daha', 'de', 'defa', 'diye', 'eğer', 'en', 'gibi', 'hem', 'hep', 'hepsi', 'her', 'hiç', 'için', 'ile', 'ise', 'kez', 'ki', 'kim', 'mı', 'mu', 'mü', 'nasıl', 'ne', 'neden', 'nerde', 'nerede', 'nereye', 'niçin', 'niye', 'o', 'sanki', 'şey', 'siz', 'şu', 'tüm', 've', 'veya', 'ya', 'yani']

mesaj = input("Yapmak istediginiz islemi giriniz: ")
mesajdf = pd.DataFrame({"metin": mesaj, "kategori":0 }, index=[42])
df = pd.concat([df, mesajdf], ignore_index=True)

for word in stopwords:
    word = " " + word + " "
    df['metin'] = df['metin'].str.replace(word, " ")

#print(df.head(3))

cv = CountVectorizer(max_features=50)



x = cv.fit_transform(df['metin']).toarray()
y = df['kategori']
tahmin = x[-1].copy()

x = x[0:-1]
y = y[0:-1]

#print(x[0])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=21, train_size=0.7)

rf = RandomForestClassifier()
model = rf.fit(x_train, y_train)
skor = model.score(x_test,y_test)
sonuc = model.predict([tahmin])

print("Sonuc: ", sonuc, "Skor: ", skor)

