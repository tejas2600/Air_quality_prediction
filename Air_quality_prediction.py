
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import  svm


df = pd.read_csv("c:/Users/Tejas/Downloads/air_quality.csv")


df.wind_direction[df.wind_direction == 'NNW'] = 1
df.wind_direction[df.wind_direction == 'N'] = 2
df.wind_direction[df.wind_direction == 'NW'] = 3
df.wind_direction[df.wind_direction == 'NNE'] = 4
df.wind_direction[df.wind_direction == 'ENE'] = 5
df.wind_direction[df.wind_direction == 'E'] = 6
df.wind_direction[df.wind_direction == 'W'] = 7
df.wind_direction[df.wind_direction == 'SSW'] = 8
df.wind_direction[df.wind_direction == 'NE'] = 9
df.wind_direction[df.wind_direction == 'WSW'] = 10
df.wind_direction[df.wind_direction == 'SE'] = 11
df.wind_direction[df.wind_direction == 'WNW'] = 12
df.wind_direction[df.wind_direction == 'SSE'] = 13
df.wind_direction[df.wind_direction == 'ESE'] = 14
df.wind_direction[df.wind_direction == 'S'] = 15
df.wind_direction[df.wind_direction == 'SW'] = 16




df['PM2.5_1'] = df['PM2.5'].shift(periods=1)
df['temperature_1'] = df.temperature.shift(periods=1)
df['pressure_1'] = df.pressure.shift(periods=1)
df['rain_1'] = df.rain.shift(periods=1)
df['wind_direction_1'] = df.wind_direction.shift(periods=1)
df['wind_speed_1'] = df.wind_speed.shift(periods=1)

df['PM2.5_2'] = df['PM2.5'].shift(periods=2)
df['temperature_2'] = df.temperature.shift(periods=2)
df['pressure_2'] = df.pressure.shift(periods=2)
df['rain_2'] = df.rain.shift(periods=2)
df['wind_direction_2'] = df.wind_direction.shift(periods=2)
df['wind_speed_2'] = df.wind_speed.shift(periods=2)

df['PM2.5_3'] = df['PM2.5'].shift(periods=3)
df['temperature_3'] = df.temperature.shift(periods=3)
df['pressure_3'] = df.pressure.shift(periods=3)
df['rain_3'] = df.rain.shift(periods=3)
df['wind_direction_3'] = df.wind_direction.shift(periods=3)
df['wind_speed_3'] = df.wind_speed.shift(periods=3)




prediction =  df[df["PM2.5"]==-1]
df = df.drop(prediction.index)

df.dropna(inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.drop(['PM2.5'], axis=1), df["PM2.5"])

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)

confidence1 = clf.score(X_train, y_train)
print(confidence1)

prediction.fillna(prediction.mean(),inplace=True)

y_predicted = clf.predict(prediction.drop(['PM2.5'], axis=1))
prediction['PM2.5'] = y_predicted

prediction=prediction.drop(['temperature_1','temperature_2','temperature_3','pressure_1','pressure_2','pressure_3','wind_direction_1','wind_direction_2','wind_direction_3','wind_speed_1','wind_speed_2','wind_speed_3','rain_1','rain_2','rain_3','PM2.5_1','PM2.5_2','PM2.5_3'],axis=1)



prediction.to_csv("c:/Users/Tejas/Downloads/Quality_PM2.5_predicted.csv",index=False)
