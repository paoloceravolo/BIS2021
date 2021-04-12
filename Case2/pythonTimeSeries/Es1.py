import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import seaborn as sns
import antropy as ant # https://github.com/raphaelvallat/antropy


from statsmodels.tsa.seasonal import seasonal_decompose


df = pd.read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', parse_dates=['date'], index_col='date')

#print(df)

## Visualize entire series 
# Draw Plot
def plot_df(df, x, y, title="", xlabel='Date', ylabel='Value', dpi=100):
    plt.figure(figsize=(16,5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    #plt.show()

plot_df(df, x=df.index, y=df.value, title='Monthly anti-diabetic drug sales in Australia from 1992 to 2008.')    

df.reset_index(inplace=True)
# extract year and mont from the data value
df['year'] = [d.year for d in df.date]
df['month'] = [d.strftime('%b') for d in df.date]
years = df['year'].unique()

# Prep Colors
#np.random.seed(100)
mycolors = np.random.choice(list(mpl.colors.XKCD_COLORS.keys()), len(years), replace=False)

# Draw Plot
plt.figure(figsize=(16,12), dpi= 80)
for i, y in enumerate(years):
    if i > 0:        
        plt.plot('month', 'value', data=df.loc[df.year==y, :], color=mycolors[i], label=y)
        plt.text(df.loc[df.year==y, :].shape[0]-.9, df.loc[df.year==y, 'value'][-1:].values[0], y, fontsize=12, color=mycolors[i])

# Decoration
plt.gca().set(xlim=(-0.3, 11), ylim=(2, 30), ylabel='$Drug Sales$', xlabel='$Month$')
plt.yticks(fontsize=12, alpha=.7)
plt.title("Seasonal Plot of Drug Sales Time Series", fontsize=20)
#plt.show()

## Visualize by boxplot 

# Draw Plot
fig, axes = plt.subplots(1, 2, figsize=(20,7), dpi= 80)
sns.boxplot(x='year', y='value', data=df, ax=axes[0])
sns.boxplot(x='month', y='value', data=df.loc[~df.year.isin([1991, 2008]), :])

# Set Title
axes[0].set_title('Year-wise Box Plot\n(The Trend)', fontsize=18); 
axes[1].set_title('Month-wise Box Plot\n(The Seasonality)', fontsize=18)
#plt.show()


# Extract Trends

# Set date as index
df = df.set_index('date')

# Multiplicative Decomposition 
result_mul = seasonal_decompose(df['value'], model='multiplicative', extrapolate_trend='freq')

#print(result_mul)

# Additive Decomposition
result_add = seasonal_decompose(df['value'], model='additive', extrapolate_trend='freq')

#print(result_add)

# Plot
plt.rcParams.update({'figure.figsize': (10,10)})
result_mul.plot().suptitle('Multiplicative Decompose', fontsize=22)
result_add.plot().suptitle('Additive Decompose', fontsize=22)
#plt.show()

# Actual Values = Product of (Seasonal * Trend * Resid)
df_reconstructed = pd.concat([result_mul.seasonal, result_mul.trend, result_mul.resid, result_mul.observed], axis=1)
df_reconstructed.columns = ['seas', 'trend', 'resid', 'actual_values']
print(df)
print(df_reconstructed)

# Using statmodels: Subtracting the Trend Component.

detrended = df.value.values - result_mul.trend
plt.plot(detrended)
plt.title('Drug Sales detrended by subtracting the trend component', fontsize=16)
#plt.show()

# Permutaiton Entropy
print('Permutaiton Entropy', ant.perm_entropy(df['value'], normalize=True))

# Autocorrelation

print('Autocorrelation with lag 2', df['value'].autocorr(lag=2))
print('Autocorrelation with lag 4', df['value'].autocorr(lag=4))
print('Autocorrelation with lag 8', df['value'].autocorr(lag=8))
print('Autocorrelation with lag 16', df['value'].autocorr(lag=16))
print('Autocorrelation with lag 32', df['value'].autocorr(lag=32))

from statsmodels.graphics import tsaplots

# Display the autocorrelation plot of your time series
fig = tsaplots.plot_acf(df['value'], lags=24)
#plt.show()

## Discretize the time series
#https://github.com/seninp/saxpy
# SAX

from saxpy.znorm import znorm
from saxpy.sax import ts_to_string
from saxpy.alphabet import cuts_for_asize

#print(ts_to_string(znorm(df['value']), cuts_for_asize(5)))

# SAX conversion via sliding window
# the result is represented as a data structure of resulting words and their respective positions on time series:

from saxpy.hotsax import find_discords_hotsax
from saxpy.paa import paa
from saxpy.sax import sax_via_window
sax_win = sax_via_window(df['value'], win_size=6, paa_size=6, alphabet_size=3, nr_strategy=None, z_threshold=0.01)
#print(sax_win)
#print([x for x in sax_win]) #list of sequences

from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX


model = ARIMA(df.value, order=(4,2,3))
model_fit = model.fit(disp=0)
#print(model_fit.summary())
#print(model_fit.forecast(1))

# Create Training and Test
train = df.value[:180]
test = df.value[180:]

fc, se, conf = model_fit.forecast(24, alpha=0.05) 

#print(fc)

# Make as pandas series
fc_series = pd.Series(fc, index=test.index)
lower_series = pd.Series(conf[:, 0], index=test.index)
upper_series = pd.Series(conf[:, 1], index=test.index)

plt.figure(figsize=(12,5), dpi=100)
plt.plot(train, label='training')
plt.plot(test, label='actual')
plt.plot(fc_series, label='ARIMA forecast')
plt.fill_between(lower_series.index, lower_series, upper_series, 
                 color='k', alpha=.15)
plt.title('Forecast vs Actuals')
plt.legend(loc='upper left', fontsize=8)
#plt.show()

from sklearn.metrics import mean_absolute_error

mae_ar = mean_absolute_error(test, fc_series)
print('ARIMA MAE: %.3f' % mae_ar)



































