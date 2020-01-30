#Import Libraries
import pandas as pd
import numpy as np
import time
import datetime
from datetime import datetime
import statsmodels.api as sm
import matplotlib .pyplot as plt
%matplotlib inline

import warnings
warnings.filterwarnings("ignore") # specify to ignore warning messages,

from pmdarima import auto_arima
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

from statsmodels.tsa.stattools import ccovf,ccf,periodogram
from statsmodels.tsa.stattools import adfuller,kpss,coint,bds,q_stat,grangercausalitytests,levinson_durbin
from statsmodels.tools.eval_measures import mse, rmse, meanabs
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose

#Load data to pandas
df = pd.read_csv('2gTraffic_MLI058MG1.csv')
df.head()

#count number of observation
df.count()

#show last five observations
df.tail()

#create index
index = pd.date_range('2018-01-01', periods=107, freq='W')
index

#set index df with above index
df.set_index(index, inplace=True)
df.index

#add name 'Date' to index
df.index.name='Date'

#summary statistic
df.describe()


#plot grafik traffic
df['volume_tch_traffic_erl'].plot(figsize=(12,5), legend=True, title='Traffic 2G weekly').autoscale(axis='x', tight=True)

#calculate 4week average
df['4Week-average'] = df['volume_tch_traffic_erl'].rolling(window=4).mean()
df[['volume_tch_traffic_erl', '4Week-average']].plot(figsize=(12,5), title ='Traffic 2G').autoscale(axis='x', tight=True)

#Lets plot decompose to see a trend, seasonal and residual of data
result = seasonal_decompose(df['volume_tch_traffic_erl'], model='multiplicative', freq=1)
result.plot();

#Lets plot decompose to see a trend, seasonal and residual of data
result = seasonal_decompose(df['volume_tch_traffic_erl'], model='multiplicative')
result.plot();

#Lets plot decompose to see a trend, seasonal and residual of data
result = seasonal_decompose(df['volume_tch_traffic_erl'], model='multiplicative', freq=36)
result.plot();

#Lets plot decompose to see a trend, seasonal and residual of data
result = seasonal_decompose(df['volume_tch_traffic_erl'], model='add', freq=4)
result.plot();

#Lets plot decompose to see a trend, seasonal and residual of data
result = seasonal_decompose(df['volume_tch_traffic_erl'], model='add')
result.plot();

#split data into train and test
train = df.iloc[:89]
test = df.iloc[89:]

#Holt WInter Model
#create model exponential smoothing
modelExpo=ExponentialSmoothing(train['volume_tch_traffic_erl'], trend='add', seasonal='mul', seasonal_periods=4).fit()

#make prediction
test_predictions=modelExpo.forecast(len(test))

#show result of prediction
test_predictions

#plot train and test
train['volume_tch_traffic_erl'].plot(legend=True, label='Train')
test['volume_tch_traffic_erl'].plot(legend=True, label='Test', figsize=(12,5))

#plot the prediction and actual data
train['volume_tch_traffic_erl'].plot(legend=True, label='Train')
test['volume_tch_traffic_erl'].plot(legend=True, label='Test', figsize=(12,5), title = '2G Traffic Analysis & Prediction')
test_predictions.plot(legend=True, label='Predictions')

#plot prediction and actual data(test data)
test['volume_tch_traffic_erl'].plot(legend=True, label='Test', figsize=(12,5), title = '2G Traffic Analysis & Prediction')
test_predictions.plot(legend=True, label='Predictions')

# model evaluation with mean absolute error
mean_absolute_error(test['volume_tch_traffic_erl'], test_predictions)

#calculate mean square error
error = mean_squared_error(test['volume_tch_traffic_erl'], test_predictions)
print(f'Holt Winter MSE Error: {error}')

#calculate rmse
rmse = np.sqrt(mean_squared_error(test['volume_tch_traffic_erl'], test_predictions))
print(f'Holt Winter RMSE: {rmse}')

#check summary statistic from data test
test['volume_tch_traffic_erl'].describe()

#ARIMA
#Test Stationary with Dickey-Fuller

#Check stationary data with Dickey Fuller
test_stat=adfuller(df['volume_tch_traffic_erl'],autolag='AIC')
test_stat

print('Augmented Dickey-Fuller Test on Traffic 2G data')

testout = pd.Series(test_stat[0:4],index=['ADF test statistic','p-value','# lags used','# observations'])

for key,val in test_stat[4].items():
    testout[f'critical value ({key})']=val
print(testout)

#plot original data and its autocorrelation
df['volume_tch_traffic_erl'].plot(legend=True, title='Original data traffic')
#plt.subplot(122)
plot_acf(df['volume_tch_traffic_erl'])
plt.show()

#plot first differencing and its autocorrelation
#calculate first differencing
diff1=df['volume_tch_traffic_erl'].diff()

diff1.plot(legend=True, title='First differencing traffic')
#plt.subplot(122)
plot_acf(diff1.dropna())
plt.show()


#plot second differencing and its autocorrelation
#calculate first differencing
diff2=diff1.diff()

diff2.plot(legend=True, title='Second differencing traffic')
#plt.subplot(122)
plot_acf(diff2.dropna())
plt.show()


#Test Dickey Fuller at the first differencing
test_stat=adfuller(diff1.dropna(), autolag='AIC')
print('Augmented Dickey-Fuller Test on the first difference 2G Traffic')

testout=pd.Series(test_stat[0:4], index=['ADF test statistic','p-value','# lags used','# observations'])

for key, val in test_stat[4].items():
    testout[f'critical value ({key})']=val
print(testout)


#plot the time series first differencing with Partial AutoCorrelation
#plot first differencing
diff1.plot()

#plot partial autocorrelation
plot_pacf(diff1.dropna());

#build ARIMA model with all data
model = ARIMA(df['volume_tch_traffic_erl'], order=(0,1,1))

#fit the model
result=model.fit()

#print the summary model
print(result.summary())

# Plot actual vs forecast
result.plot_predict(dynamic=False) #dynamic false means the in-sample lagged values are used for prediction. 
plt.show()

#build model ARIMA on data train
model = ARIMA(train['volume_tch_traffic_erl'], order=(0,1,1))

#fit model
result = model.fit()

#print summary model
print(result.summary())

#make prediction
start = len(train)
end = len(train)+len(test)-1
predictions = result.predict(start=start, end=end, dynamic=False, typ='levels').rename('ARIMA(0,1,1) Predictions')

# Compare predictions to expected values
for i in range(len(predictions)):
    print(f"predicted={predictions[i]:<11.10}, expected={test['volume_tch_traffic_erl'][i]}")
    
#plot predictions against real data
title ='Traffic 2G Prediction'
ylabel = 'tch_traffic erlang'
xlabel=''

ax= test['volume_tch_traffic_erl'].plot(legend=True, figsize=(12,6), title=title)
predictions.plot(legend=True)
ax.autoscale(axis='x', tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);

#Evaluate Model
#calculate MSE 
#calculate MSE 
error = mean_squared_error(test['volume_tch_traffic_erl'], predictions)
print(f'ARIMA(0,1,1) MSE Error: {error:11.10}')

#calculate RMSE
error = np.sqrt(mean_squared_error(test['volume_tch_traffic_erl'], predictions))
print(f'ARIMA(0,1,1) RMSE Error:{error:11.10}')

#Auto Arima
auto_arima(df['volume_tch_traffic_erl'], seasonal=False, m=12).summary()

# Result of auto arima also same order (0,1,1)

#try auto arima with seasonal true
auto_arima(df['volume_tch_traffic_erl'], seasonal=True, m=12).summary()

#build model SARIMA with order from result auto arima
SARIMA_model2 = SARIMAX(train['volume_tch_traffic_erl'], order=(0,1,1), seasonal_order=(1,0,2,12))
result2 = SARIMA_model2.fit()
result2.summary()

# Obtain predicted values
start=len(train)
end=len(train)+len(test)-1
predictions2 = result2.predict(start=start, end=end, dynamic=False, typ='levels').rename('SARIMA(0,1,1)x(1,0,2,12) Predictions')

#show the prediction
predictions2


# Compare predictions to expected values
for i in range(len(predictions2)):
    print(f"predicted={predictions2[i]:<11.10}, expected={test['volume_tch_traffic_erl'][i]}")
    
#plot predictions against real data
title ='Traffic 2G Prediction'
ylabel = 'tch_traffic erlang'
xlabel=''

ax= test['volume_tch_traffic_erl'].plot(legend=True, figsize=(12,6), title=title)
predictions2.plot(legend=True)
ax.autoscale(axis='x', tight=True)
ax.set(xlabel=xlabel, ylabel=ylabel);

# Evaluate Model SARIMAX(0,1,1)x(1,0,2,12)
#calculate mean square error
error = mean_squared_error(test['volume_tch_traffic_erl'], predictions2)
print(f'SARIMAX(0,1,1)x(1,0,2,12) MSE Error: {error}')

#calculate rmse
rmse = np.sqrt(mean_squared_error(test['volume_tch_traffic_erl'], predictions2))
print(f'SARIMAX(0,1,1)x(1,0,2,12) RMSE: {rmse}')

#show summary statistic of data test
test['volume_tch_traffic_erl'].describe()



