#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pmdarima --q')
get_ipython().system('pip install prophet --q')


# In[2]:


import warnings
warnings.filterwarnings('ignore')
import pandas as pd
df = pd.read_excel('Oil_Prices.xlsx')
df.head()


# In[3]:


df.columns = df.columns.str.strip()
df.columns


# In[4]:


try:
    column_name = 'Price'  # Replace with the actual column name
    column_data = df.loc[:, column_name]
except KeyError:
    print("Column does not exist in the DataFrame.")


# In[5]:


# Descriptive Analysis


# In[6]:


df.shape


# In[7]:


df.size


# In[8]:


df.dtypes


# In[9]:


df.info()


# In[10]:


df.describe()
# The average price of the oil is 50.253 d/b.
# The variation in the price is upto 29.35 d/b.
# The minimum price of the oil is -37.63 d/b.
# The maximum price of the oil is 145.29 d/b.


# In[11]:


mean_value = df[df['Price']>=0]['Price'].mean()
mean_value


# In[12]:


df.loc[df['Price']<0,'Price'] = mean_value


# In[13]:


df.describe()


# In[14]:


df.nunique()


# In[15]:


# Data Exploration


# In[16]:


#df.isna().sum()
df.isnull().sum()


# In[17]:


df['Price'].plot(figsize=(16,6))


# In[18]:


# Feature Engineering


# In[19]:


# Marking the date as date time index for dataframe


# In[20]:


oil_price = df.copy()
oil_price.set_index('Date',inplace=True)
oil_price.index.year


# In[21]:


oil_price.head()


# In[22]:


# Separating the month and year into a separate column for visualization and removing meaningful insight of the data.


# In[23]:


df['Date'] = pd.to_datetime(df['Date'], format='%b-%y')
df['month'] = df.Date.dt.strftime('%b') # for month extraction
df['Year'] = df.Date.dt.strftime('%y') # for year extracton


# In[24]:


df.head()


# In[25]:


# Data Visualization


# In[26]:


import seaborn as sns
import matplotlib.pyplot as plt
plt.figure(figsize=(14,8))
heatmap_y_month = pd.pivot_table(data=df,values="Price",index='Year',columns="month",fill_value=0)
sns.heatmap(heatmap_y_month,annot=True,fmt="g") #fmt is format of the grid values
plt.show()


# In[27]:


# As showed that the light colour will be the maximum prices as 134d/b and 133d/b
# whereas the dark or black colur refers to the minimum prices.


# In[28]:


# Line Plot


# In[29]:


oil_price['Price'].plot(figsize=(12,6),grid=True)
plt.title('variation of oil prices')
plt.show()


# In[30]:


#By observing above line plot we can notice that the trend is increasing and decreasing.Hence,trend is not constant.
#Variance is also not constant.
#And also time series is not stationary.


# In[31]:


# Data Visualization Through Histograms and Kde plots


# In[32]:


oil_price.skew()


# In[33]:


oil_price['Price'].hist()


# In[34]:


# By noticing the above histogram, it can be noticed that it is a positively skewed histogram or right skewed histogram.


# In[35]:


oil_price['Price'].plot(kind='kde')
plt.title('Density plot of price')
plt.show()


# In[36]:


# Yearly Price Analysis


# In[37]:


plt.figure(figsize=(12,6))
sns.lineplot(x='Year',y='Price',data=df)


# In[38]:


# Boxplot Of Each Year With Monthly Intervals


# In[39]:


plt.figure(figsize=(14,6))
plt.subplot(211)
sns.boxplot(x='month',y='Price',data=df)
plt.subplot(212)
sns.boxplot(x='Year',y='Price',data=df)
plt.show()


# In[40]:


# Calculating The Interquartile Range


# In[41]:


Q1 = df['Price'].quantile(0.25)
Q3 = df['Price'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1-1.5*IQR
upper_bound = Q3-1.5*IQR

df_no_outliers = df[(df['Price'] >= lower_bound) | (df['Price'] <= upper_bound)]
print(df_no_outliers)


# In[42]:


# Lag Plot


# In[43]:


# create a scatter plot
from pandas.plotting import lag_plot
for i in [15,30,45,60,90]:
    lag_plot(df['Price'], lag=i)
    plt.show()


# In[44]:


# Time Series Decomposition


# In[45]:


from statsmodels.tsa.seasonal import seasonal_decompose
decompose_ts = seasonal_decompose(df['Price'], period=12)
with plt.rc_context():
    plt.rc('figure',figsize=(14,6))
    decompose_ts.plot()
    plt.show()


# In[46]:


# ACF And PACF PLots

# Auto Correlation Function


# In[47]:



import statsmodels.graphics.tsaplots as tsaplots
with plt.rc_context():
    for i in [30,45,60,90]:
        plt.rc('figure',figsize=(14,6))
        tsaplots.plot_acf(df['Price'],lags=i)
        plt.show()


# In[48]:


# Partial Autocorrelation Function(PACF)


# In[49]:


with plt.rc_context():
    for i in [30,45,60,90]:
        plt.rc('figure',figsize=(14,6))
        tsaplots.plot_pacf(df['Price'], lags=i)
        plt.show()


# In[50]:


plt.figure(figsize=(14,6))
df['Price'].plot(label='org')
for i in range(4,40,4):
    df['Price'].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')


# In[51]:



df.shape


# In[52]:


df = df.drop(['month','Year'], axis=1)


# In[53]:


df.set_index('Date',inplace=True)
df.values


# In[54]:


# Stationarity Checking


# In[55]:


# ADFuller test (Augumented Dickey Fuller Test)
# If Test statistic < Critical Value and p-value < 0.05 – then series is stationary.
from statsmodels.tsa.stattools import adfuller
def adf_test(timeseries):
    print('Results of adfuller test:')
    print('------------------------')
    adftest = adfuller(timeseries)
    adf_output = pd.Series(adftest[0:4], index=['Test statistic','p-value','#lags used','Number of Observations used'])
    for key, value in adftest[4].items():
        adf_output['Critical Value (%s)'%key] = value
        print(adf_output)
adf_test(df.values)


# In[56]:


# time series is not stationary because p-value is greater than level of significance(0.05).


# In[57]:


diff_df = df.diff()
diff_df.head()


# In[58]:


diff_df.dropna(inplace=True)


# In[59]:


diff_df.plot(grid=True)


# In[60]:


adf_test(diff_df.values)


# In[61]:


# time series is stationary because p value is less the than level of significance(0.05).


# In[62]:


tsaplots.plot_acf(diff_df);
tsaplots.plot_pacf(diff_df);
plt.show()


# In[63]:


# From ACF, we got q values as 1,4,6,22 and 33.
# From PACF, we got p values as 1,4,6,22,26 and 34.


# In[64]:


df.reset_index('Date')


# In[65]:


train_df = df.head(5000)
test_df = df.tail(3553)


# In[66]:


train_df.shape


# In[67]:


test_df.shape


# In[68]:


# Data Modelling


# In[69]:


# Single Exponential Smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
single_exp = SimpleExpSmoothing(train_df).fit(smoothing_level=0.3)
single_exp_trained_pred = single_exp.fittedvalues
single_exp_test_pred = single_exp.forecast(3553)


# In[70]:


pip install tensorflow


# In[71]:


from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
print('Train RMSE:',mean_squared_error(train_df,single_exp_trained_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df,single_exp_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df,single_exp_trained_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df,single_exp_test_pred))


# In[72]:


# Double Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
double_exp = ExponentialSmoothing(train_df,trend=None,initialization_method='heuristic', seasonal='add',seasonal_periods=29,damped_trend=False).fit()
double_exp_train_df = double_exp.fittedvalues
double_exp_test_df = double_exp.forecast(3553)


# In[73]:


print('Train RMSE:',mean_squared_error(train_df,double_exp_train_df)**0.5)
print('Test RMSE:',mean_squared_error(test_df,double_exp_test_df)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df,double_exp_train_df))
print('Test MAPE:',mean_absolute_percentage_error(test_df,double_exp_test_df))


# In[74]:


# Triple Exponential Smoothing
triple_exp = ExponentialSmoothing(train_df['Price'],trend='add',initialization_method='heuristic',seasonal='add',seasonal_periods=28,damped_trend=True).fit()
triple_exp_pred_train = triple_exp.fittedvalues
triple_exp_pred_test = triple_exp.forecast(3553)


# In[75]:


print('Train RMSE:',mean_squared_error(train_df,triple_exp_pred_train)**0.5)
print('Test RMSE:',mean_squared_error(test_df,triple_exp_pred_test)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df,triple_exp_pred_train))
print('Test MAPE:',mean_absolute_percentage_error(test_df,triple_exp_pred_test))


# In[76]:


# Arima Model


# In[77]:


from statsmodels.tsa.arima.model import ARIMA
ar = ARIMA(train_df,order=(6,1,6)).fit()
ar_train_pred = ar.fittedvalues
ar_test_pred = ar.forecast(3553)


# In[78]:


print('Train RMSE:',mean_squared_error(train_df,ar_train_pred)**0.5)
print('Test RMSE:',mean_squared_error(test_df,ar_test_pred)**0.5)
print('Train MAPE:',mean_absolute_percentage_error(train_df,ar_train_pred))
print('Test MAPE:',mean_absolute_percentage_error(test_df,ar_test_pred))


# In[79]:


# Auto Arima Model


# In[80]:


from pmdarima.arima import auto_arima
stepwise_fit = auto_arima(df['Price'], trace=True, suppress_warnings=True)


# In[81]:


stepwise_fit.summary()


# In[82]:


from statsmodels.tsa.arima_model import ARIMA


# In[83]:


df = df.sort_values('Date',ascending=True)
df.tail()


# In[84]:


print(df.shape)
train = df.iloc[:5000]
test = df.iloc[5000:]
print(train.shape)
print(test.shape)


# In[85]:


# Train the model
from statsmodels.tsa.statespace.sarimax import SARIMAX
model = SARIMAX(train['Price'], order=(3,1,2))
model = model.fit()
model.summary()


# In[86]:


start = len(train)
end = len(train)+len(test) - 1
pred = model.predict(start=start,end=end,type='levels')
print(pred)
# pred.index = df.index[start:end+1]


# In[87]:


test['Price'].mean()


# In[88]:


from sklearn.metrics import mean_squared_error
from math import sqrt
rmse = sqrt(mean_squared_error(pred,test['Price']))
print(rmse)


# In[89]:


model2 = SARIMAX(df['Price'],order=(3,1,2))
model2 = model2.fit()
df.tail()


# In[90]:


# For Future Dates
import pandas as pd
index_future_dates = pd.date_range(start='2023-07-26',end='2023-08-25')
print(index_future_dates)
pred = model2.predict(start=len(df),end=len(df)+30,type='levels').rename('ARIMA Predictions')
#print(comp_pred)
pred.index=index_future_dates
print(pred)


# In[91]:


pred.plot(figsize=(16,6), legend=True)


# In[92]:


# Prophet Model


# In[93]:


get_ipython().system('pip install pystan')


# In[94]:


from prophet import Prophet
from prophet.plot import plot_plotly,plot_components_plotly


# In[95]:


df.reset_index('Date',inplace=True)


# In[96]:


df.columns=['ds','y']
df.tail()


# In[97]:


# plotting the data
df.plot(x='ds', y='y', figsize=(16,6))


# In[98]:


len(df)


# In[99]:


train = df.iloc[:len(df)-365]
test = df.iloc[len(df)-365:]


# In[100]:


print(train.shape,test.shape)


# In[101]:


m = Prophet()
m.fit(train)
future = m.make_future_dataframe(periods=365)
forecast = m.predict(future)


# In[102]:


forecast


# In[103]:


forecast[['ds','yhat','yhat_lower','yhat_upper']]


# In[104]:


df


# In[105]:


plot_plotly(m,forecast)


# In[106]:


plot_components_plotly(m,forecast)


# In[107]:


from statsmodels.tools.eval_measures import rmse
predictions = forecast.iloc[-365:]['yhat']


# In[108]:


print('rmse between actual and predicted values:',rmse(predictions,test['y']))
print('mean value of test Dataset:',test['y'].mean())


# # Forecasting Methods

# In[109]:


get_ipython().system('pip install forecast_tools')


# In[110]:


df.columns=['Date','Price']
df.set_index('Date',inplace=True)
df.head()


# In[111]:


def preds_as_series(data, preds):
  '''
  Helper function for plotting predictions.
  converts a numpy array of predictions to a
  pandas. DataFrame with datetimeindex

  Parameters
  ------
  pred - numpy.array,vector of predictions
  start - start date of the time series
  freq - the frequency of the time series e.g 'MS' or 'D'
  '''
  start = pd.date_range(start=data.index.max(), periods=2, freq=data.index.freq).max()
  idx = pd.date_range(start=start, periods=len(preds), freq=data.index.freq)
  return pd.DataFrame(preds, index=idx)


# In[112]:


df.plot(figsize=(12,4))


# In[113]:


# Naive Forecasting Method
from forecast_tools.baseline import Naive1
nf1 = Naive1()
nf1.fit(df)
nf1_preds = nf1.predict(horizon=1553)


# In[114]:


nf1_preds


# In[115]:


ax = df.plot(figsize=(12,4), legend=True)
nf1.fittedvalues.plot(ax=ax, linestyle='-.')
preds_as_series(df, nf1_preds).plot(ax=ax)
ax.legend(['train', 'Naive1'])


# In[116]:


# Seasonal Naive
from forecast_tools.baseline import SNaive
snf = SNaive(period=7)
snf.fit(df)
snf_preds = snf.predict(horizon=1553)
snf_preds


# In[117]:


ax = df.plot(figsize=(12,4))
snf.fittedvalues.plot(ax=ax, linestyle='-.')
preds_as_series(df,snf_preds).plot(ax=ax)
ax.legend(['train','Fitted Model', 'SNaive Forecast'])


# In[120]:


get_ipython().system('pip install sktime')


# In[122]:


def plot_prediction_intervals(train, preds, intervals, test=None):
  '''
  Helper function to plot training data, point preds
  and 2 sets of prediction intervals

  assume 2 sets of PIs are provided!
  '''
  ax = train.plot(figsize=(12,4))

  mean = preds_as_series(train, preds)
  intervals_80 = preds_as_series(train, intervals[0])
  intervals_90 = preds_as_series(train, intervals[1])

  mean.plot(ax=ax, label='point forecast')
  ax.fill_between(intervals_80.index, mean[0], intervals_80[1], alpha=0.2, label='80% PI', color='yellow');
  ax.fill_between(intervals_80.index,mean[0], intervals_80[0], alpha=0.2, label='80% PI', color='yellow');
  ax.fill_between(intervals_80.index,intervals_80[1], intervals_90[1], alpha=0.2, label='90% PI', color='purple');
  ax.fill_between(intervals_80.index,intervals_80[0], intervals_90[0], alpha=0.2, label='90% PI', color='purple');

  if test is None:
    ax.legend(['train','point forecast', '80% PI', '_ignore', '_ignore', '90% PI'], loc=2)
  else:
    test.plot(ax=ax, color='black')
    ax.legend(['train','point forecast', 'Test', '80% PI','_ignore', '_ignore', '90% PI'], loc=2)


# In[124]:


# Measuring Point Forecast Error
# A Basic Train Test Split
train_length = df.shape[0] - 1553
train, test = df.iloc[:train_length], df.iloc[train_length:]


# In[125]:


train.shape, test.shape


# In[126]:


snf = SNaive(period=7)
preds = snf.fit_predict(train, horizon=1553)
preds


# In[127]:


from forecast_tools.metrics import forecast_errors
forecast_errors(test, preds)


# In[128]:


from forecast_tools.baseline import Naive1
nf1 = Naive1()
nf1.fit(df)
nf1_preds = nf1.predict(horizon=1553)


# In[129]:


forecast_errors(test,nf1_preds)


# # LSTM(Long Short Term Memory) Model

# In[130]:


df.head()


# In[131]:


df.plot(figsize=(16,6))


# In[132]:


from statsmodels.tsa.seasonal import seasonal_decompose
results = seasonal_decompose(df['Price'], period=12)
results.plot()


# In[133]:


len(df)


# In[134]:


df.describe()


# In[135]:


# Creating a simple moving average for 7 and 21 days
df['ma7'] = df.Price.rolling(window=7).mean()
df['ma21'] = df.Price.rolling(window=21).mean()

# Creating the EMA  i.e, EMA --> Exponential Moving Average
df['ema12'] = df.Price.ewm(span=12).mean().fillna(0)
df['ema26'] = df.Price.ewm(span=26).mean().fillna(0)
df['macd'] = df.ema12 - df.ema26

#The variables below are used for Bollinger Bands.
window=21
no_std = 2
rolling_mean = df.Price.rolling(window).mean()
rolling_std = df.Price.rolling(window).std()
df['bollinger_low'] = (rolling_mean - (rolling_std * no_std)).fillna(0)
df['bollinger_high'] = (rolling_mean + (rolling_std * no_std)).fillna(0)
df['ema'] = df.Price.ewm(com=0.5).mean()
df['momentum'] =  df.Price - 1

df.head()


# In[136]:


from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
dataset = mms.fit_transform(df['Price'].values.reshape(-1,1))
dataset[0:10]


# In[137]:


# Split Into Train And Test Sets
train = int(len(dataset)*0.7)
test = len(dataset)-train
train, test = dataset[0:train,:], dataset[train:len(dataset),:]
f'Dataset size: {len(df)} >> Train length: {len(train)} || Test Length: {len(test)}'


# In[138]:


# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[139]:


import numpy as np
x_train, y_train = create_dataset(train, look_back=15)
x_test, y_test = create_dataset(test, look_back=15)
f'X_train: {x_train.shape} || y_train: {y_train.shape} || X_test: {x_test.shape} || y_test: {y_test.shape}'


# In[140]:


x_train = np.reshape(x_train, (x_train.shape[0], 1, x_train.shape[1]))
x_test = np.reshape(x_test, (x_test.shape[0], 1, x_test.shape[1]))
f'X_train: {x_train.shape} || y_train: {y_train.shape} || X_test: {x_test.shape} || y_test: {y_test.shape}'


# In[141]:


get_ipython().system('pip install tensorflow')


# In[142]:


import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.models import load_model
from keras.layers import LSTM
import keras
import requests
import h5py
import os


# In[143]:


# create and fit the LSTM network
look_back = 15
model = Sequential()
model.add(LSTM(20,activation = 'relu',input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=20, batch_size=1, verbose=2)


# In[144]:


import math
from sklearn.metrics import mean_squared_error
trainPredict = model.predict(x_train)
testPredict = model.predict(x_test)
# invert predictions
trainPredict = mms.inverse_transform(trainPredict)
trainY = mms.inverse_transform([y_train])
testPredict = mms.inverse_transform(testPredict)
testY = mms.inverse_transform([y_test])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train RMSE Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE Score: %.2f RMSE' % (testScore))


# In[145]:


# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.figure(figsize=(20,5))
plt.plot(trainPredictPlot, color='black', label='Train data')
plt.plot(testPredictPlot, color='blue', label='Prediction',)
plt.plot(mms.inverse_transform(dataset),label='baseline', alpha=0.4, linewidth=5)
plt.title('Daily historical Brent Oil Prices available on the U.S. Energy Information Admin', fontsize=14)
plt.ylabel('Dollars per Barrel')
plt.legend()
plt.show()


# In[146]:


# Model Comparision According Through RMSE Values


# In[147]:


model_comparision = pd.DataFrame([['Single Exponential Model',15.376],['Double Exponential Model',16.375],['Triple Exponential Model',16.382],['Arima model',15.225],['Sarima Model',22.305],['Naive Forecast model',23.057],['Seasonal Forecast model',25.489],['Prophet model',23.568],['LSTM model',1.97]],columns=['Model','RMSE'])


# In[148]:


model_comparision


# In[149]:


model_comparision.set_index('Model',inplace=True)


# In[150]:


model_comparision.sort_values('RMSE',ascending=True)


# In[151]:


# Observation: In the above models, LSTM model have less rmse value and has performed best with minimum rmse and mape and thus it can be best suitable for prediction.


# In[152]:


# LSTM model
# from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
dataset = mms.fit_transform(df['Price'].values.reshape(-1,1))
dataset[0:10]


# In[153]:


# Convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=15):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


# In[154]:


X, Y = create_dataset(dataset, look_back=15)
X = np.reshape(X,(X.shape[0],1,X.shape[1]))


# In[155]:


# creating model
# create and fit the LSTM network
look_back = 15
model = Sequential()
model.add(LSTM(20, activation = 'relu',input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(X, Y, epochs=20, verbose=0)


# In[156]:


# Saving Model into a File


# In[157]:


# serializing model to json
model_json = model.to_json()
with open('model.json','w') as json_file:
  json_file.write(model_json)
# serialize weighs to HDF5
model.save_weights('model.h5')
print('Saved model to disk')


# In[158]:


#loading model from json file for testing purpose
# load json an
d create model
import tensorflow as tf
from keras.models import model_from_json
json_file = open('model.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# Laod weights into new model
loaded_model.load_weights('model.h5')
print('loaded model from disk')


# In[159]:


testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test RMSE Score: %.2f RMSE' % (testScore))


# In[ ]:





# In[ ]:




