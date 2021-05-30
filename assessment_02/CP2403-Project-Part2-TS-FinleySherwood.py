# This file was original a Jupyter Notebook but was copied into a python file

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.arima_model import ARIMA



sales = pd.read_csv('sales.csv')
sales.head()



sales['Week Ending Date'] = pd.to_datetime(sales['Week Ending Date'], format='%d/%m/%Y')



sales.set_index('Week Ending Date', inplace=True)
sales.head()



sales['Instant'] = pd.to_numeric(sales['Instant'])
print(sales.describe())



sales.index



sales[:'2015-07-18']



sales['2018']



plt.plot(sales)



sales['month'] = sales.index.month
sales.head()



ax = sns.boxplot(x='month', y='Instant', data=sales)



def test_stationarity(timeseries):
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()

    # Plot rolling statistics
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label = 'Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)



test_stationarity(sales['Instant'])



# Perform Dickey-Fuller test
def test_Dickey_Fuller(timeseries):
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=[
        'Test Statistic',
        'p-value',
        '#Lags Used',
        'Number of Observations Used'
    ])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)'%key] = value
    print(dfoutput)



test_Dickey_Fuller(sales['Instant'])



ts_log = np.log(sales['Instant'])
plt.plot(ts_log)



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
plt.plot(seasonal, label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.tight_layout()



ts_log_decompose = residual
ts_log_decompose.dropna(inplace=True)
test_stationarity(ts_log_decompose)



test_Dickey_Fuller(ts_log_decompose)



ts_log_diff = ts_log - ts_log.shift()
ts_log_diff.dropna(inplace=True)
lag_acf = acf(ts_log_diff, nlags=20, fft=False)
lag_pacf = pacf(ts_log_diff, nlags=20, method='ols')



# ACF
plt.subplot(121) 
plt.plot(lag_acf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.title('Autocorrelation Function')

# PACF
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0, linestyle='--', color='gray')
plt.axhline(y=-1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.axhline(y=1.96/np.sqrt(len(ts_log_diff)), linestyle='--', color='gray')
plt.title('Partial Autocorrelation Function')
plt.tight_layout()



# Ignoring warnings given from jupyter notebook as they are mostly about deprecation
import warnings
warnings.filterwarnings('ignore')



model = ARIMA(ts_log, order=(2, 1, 1))
results_ARIMA = model.fit(disp=-1)  
plt.plot(ts_log_diff)
plt.plot(results_ARIMA.fittedvalues, color='red')



predictions_ARIMA_diff = pd.Series(results_ARIMA.fittedvalues, copy=True)



predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()



predictions_ARIMA_log = pd.Series(ts_log.iloc[0], index=ts_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)



predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(sales['Instant'])
plt.plot(predictions_ARIMA)