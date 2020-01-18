# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
stock="infy"

# hyper parameters
historyBatchSize=60 #Indicates past record set used to compute label. i.e last 60 days data used to arrive at price of 61st day
epochSize=1
fitBatchSize=32



if  stock=='google':
    trainCSV='Google_Stock_Price_Train.csv'
    testCSV='Google_Stock_Price_Test.csv'
    priceColumn="Open"
    priceColumnIndex=1 # position of price column

if stock=='infy':
    trainCSV='infy_train.csv'
    testCSV='infy_test.csv'
    priceColumn="Open Price"
    priceColumnIndex=8
#INFY

print(trainCSV, testCSV,priceColumnIndex, priceColumn)

# df.ID = pd.to_numeric(df.ID, errors='coerce')
# Importing the training set
dataset_train = pd.read_csv(trainCSV)

# Force column into numberic
dataset_train[priceColumn] =pd.to_numeric(dataset_train[priceColumn], errors='coerce')


training_set = dataset_train.iloc[:, priceColumnIndex:priceColumnIndex+1].values
trainingDatasetLength = len(dataset_train.index)




# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Creating a data structure with 60 timesteps and 1 output
X_train = []
y_train = []
for i in range(historyBatchSize, trainingDatasetLength):
    X_train.append(training_set_scaled[i-historyBatchSize:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train)

# Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))



# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

# Adding a second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

# Adding a fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

# Fitting the RNN to the Training set
regressor.fit(X_train, y_train, epochs = epochSize, batch_size = fitBatchSize)



# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv(testCSV)
#INFY NSE SPecific
dataset_test[priceColumn] =pd.to_numeric(dataset_test[priceColumn], errors='coerce')


real_stock_price = dataset_test.iloc[:, priceColumnIndex:priceColumnIndex+1].values

# Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train[priceColumn], dataset_test[priceColumn]), axis = 0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - historyBatchSize:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []

futureBatchSize=len(dataset_test.index)

for i in range(historyBatchSize, historyBatchSize+futureBatchSize):
    X_test.append(inputs[i-historyBatchSize:i, 0])
X_test = np.array(X_test)
print(len(X_test),X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
print('after reshape')
print(X_test)
predicted_stock_price = regressor.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

# Visualising the results
plt.plot(real_stock_price, color = 'red', label = 'Real Stock Price')
plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted  Stock Price')
plotTitle = 'historyBatch:{}-epochs:{}-futurebatch:'.format( historyBatchSize,epochSize,futureBatchSize)
plt.title(plotTitle)
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()
