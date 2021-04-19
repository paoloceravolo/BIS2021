# forecast with random forest
from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot

# Transform a time series dataset into a supervised learning dataset
# We can restructure this time series dataset as a supervised learning problem 
# by using the value at the previous time step to predict the value at the next time-step.
# This means that methods that randomize the dataset during evaluation, like k-fold cross-validation, cannot be used.
# The function below will take a time series as a NumPy array time series with one or more columns and transform it into 
# a supervised learning problem with the specified number of inputs and outputs.
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# put it all together
	agg = concat(cols, axis=1)
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# split a univariate dataset into train/test sets, n_test is the number of observations to be used for splitting
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# fit an random forest model and make a one step prediction
def random_forest_forecast(train, testX):
	# transform list into array
	train = asarray(train)
	# split into input and output columns
	trainX, trainy = train[:, :-1], train[:, -1]
	# fit model
	model = RandomForestRegressor(n_estimators=1000)
	model.fit(trainX, trainy)
	# make a one-step prediction
	yhat = model.predict([testX])
	#print('yhat', yhat[0])
	return yhat[0]

# walk-forward validation for univariate data
# In walk-forward validation, the dataset is first split into train and test sets by selecting a cut point with n_test
def walk_forward_validation(data, n_test):
	predictions = list()
	# split dataset
	train, test = train_test_split(data, n_test)
	# seed history with training dataset
	history = [x for x in train]
	# step over each time-step in the test set
	for i in range(len(test)):
		# split test row into input and output columns
		testX, testy = test[i, :-1], test[i, -1]
		# fit model on history and make a prediction
		yhat = random_forest_forecast(history, testX)
		# store forecast in list of predictions
		predictions.append(yhat)
		# add actual observation to history for the next loop
		history.append(test[i])
		# summarize progress in the shell
		# print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# estimate prediction error
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

# load the dataset
series = read_csv('https://raw.githubusercontent.com/selva86/datasets/master/a10.csv', header=0, index_col=0)
values = series.values
# transform the time series data into supervised learning
# incresing n_in, the number of observation to be taken in input to the RF you get better predction
# with 1 the peaks are identified better, with 12 the average error is reduced, with more than 12 the error increseas again
data = series_to_supervised(values, n_in=12)
# evaluate		
# firs parameter is the dataset, the second parameter is the cut point (n_test), i.e. the number of observations used in the training stage
mae, y, yhat = walk_forward_validation(data, 150)
print('MAE: %.3f' % mae)
# plot expected vs predicted
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, label='Predicted')
pyplot.legend()
pyplot.show()