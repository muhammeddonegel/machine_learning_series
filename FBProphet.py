from prophet import Prophet
import yfinance as yf
import matplotlib.pyplot as plt


df = yf.download('BTC-USD', "2015-01-01", "2025-06-26")
#print(df)

df = df[['Close']]
df = df.reset_index()
#df.head(3)

df.columns = ['ds', 'y']
#df.head(3)

model = Prophet()
model.fit(df)

future = model.make_future_dataframe(360)
tahmin = model.predict(future)
model.plot(tahmin)

model.plot_components(tahmin)
plt.show()

