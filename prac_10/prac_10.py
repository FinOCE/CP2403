# This file contains the code of a Jupyter Notebook, where each separation of 4 line breaks is a new cell

import pandas as pd
import numpy as np
import matplotlib.pylab as plt



champagne = pd.read_csv('champagne.csv')
champagne.head()



from datetime import datetime
champagne['Month'] = pd.to_datetime(champagne['Month'])
champagne.head()



champagne.set_index('Month', inplace=True)
champagne.head()



champagne['champagne'] = pd.to_numeric(champagne['champagne'])
print(champagne['champagne'].describe())



champagne.index



champagne['1965-07-01':'1965-12-01']



champagne[:'1966-07-01']



champagne['1972']



plt.plot(champagne)



champagne['Month'] = champagne.index.month
champagne.head()



import seaborn as sns
ax = sns.boxplot(x='Month', y='champagne', data=champagne)



def test_stationarity(timeseries):
    #Determing rolling statistics
    rolmean = timeseries.rolling(window=12).mean()
    rolstd = timeseries.rolling(window=12).std()
 
    #Plot rolling statistics:
    orig = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False) 



test_stationarity(champagne)



from statsmodels.tsa.stattools import adfuller

#Perform Dickey-Fuller test:
def test_Dickey_Fuller(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key,value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)



test_Dickey_Fuller(champagne['champagne'])



ts_log = np.log(champagne['champagne'])
plt.plot(ts_log)



from statsmodels.tsa.seasonal import seasonal_decompose

decomposition = seasonal_decompose(ts_log)
trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid

plt.subplot(411)
plt.plot(ts_log, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()



ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)



test_Dickey_Fuller(ts_log_decompose)



from statsmodels.tsa.stattools import acf, pacf



ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
lag_acf = acf(ts_log_diff, nlags=20, fft=True) # Adding fft property to prevent warning
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')



#Plot ACF: 
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

#Plot PACF:
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()



from statsmodels.tsa.arima_model import ARIMA



# Ignoring warnings given from jupyter notebook as they are mostly about deprecation
import warnings
warnings.filterwarnings('ignore')



#ARIMA
model = ARIMA(ts_log, order=(1, 1, 1))  #(p,d,q)
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')



predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)



predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()



predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)



predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(champagne['champagne'])
plt.plot(predictions_ARIMA)