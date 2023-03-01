# stock-price-prediction
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math 
import seaborn as sns 
import warnings
warnings.filterwarnings('ignore')
import scipy as sp
from matplotlib import patches
from scipy.spatial import ConvexHull
sns.set_style("white")

Stocks_data = pd.read_csv('C:/Users/MANSI BHOKARDOLE/Downloads/stock_price_data.csv')
Stocks_data.head()

Stocks_data.columns = (['Date', 'Open', 'High', 'Low', 'Close', 'Adj_Close', 'Volume'])
Stocks_data.columns

Stocks_data.info()

Stocks_data.value_counts() # Value counts

Stocks_data['High'].value_counts(normalize = True) # Value counts after normalizing

Stocks_data.shape # Checking shape of the data

Stocks_data.apply(np.max)

# Checking data type of all the columns

Stocks_data.dtypes

#Suppose we want to see how observations in our sample are distributed in the context of two features Low and High. 
#To do this, we can build cross tabulation by the crosstab method.

pd.crosstab(Stocks_data["Low"], Stocks_data["High"])

#Here 0 is showing the Low prices of the stocks and 1 showing the High price of the Stocks in the market as per the given data.

pd.crosstab(Stocks_data["Low"],
            Stocks_data["High"],
            normalize = 'index')

## After normalizing the indexes

Stocks_data.pivot_table(
    ["Low", "High"],
    ["Volume"],
    aggfunc = "mean",
).head(10)

Stocks_data.ndim # dimension of the data

Stocks_data['Low'].mean() # Mean value of the stocks

Stocks_data.describe() # Statistical analysis of the data

# Checking Null Values in the data

Stocks_data.isnull().sum().sum()

#DATA VISUALIZATION

Stocks_data.plot()
plt.title("Complete dataset")
plt.show()

sns.regplot(data = Stocks_data, x = 'Low', y = 'High', logistic = False)
plt.title("reg plot of Stocks price")
plt.show()

#It is a regression plot, the regplot() function takes an argument logistic, 
#which allows you to specify whether you wish to estimate the logistic regression model for the given data using True or False values.
#But un this project we are applying many models so no need to set True here..

# Create subplots 
f, axes = plt.subplots(1, 2, figsize=(7, 5))

# Plot Histogram plot with Low column
sns.distplot(Stocks_data["Low"] , color="skyblue", ax=axes[0])

# Plot Histogram plot with High column
sns.distplot( Stocks_data["High"] , color="olive", ax=axes[1])

#The plot is just showing the density like how the Low and High values of Stock price are increasing or decreasing at the same. time

# Plotting a histogram to view how the 'Low' and 'High' feature are laid out.

sns.histplot(data=Stocks_data["Low"] , color="skyblue", label="Increasing of Stocks Price (High)")
sns.histplot(data=Stocks_data["High"] , color="purple", label="Decreasing of Stocks Price (Low)")
plt.title("Comparision between Stock prices (Low and High)")
plt.show()

#You can easily compare between the increasing/ decreasing of the Stocks (Low & High).

plt.scatter(x = Stocks_data['Low'], y = Stocks_data['High'])
plt.xlabel("Stocks price of (Low) category")
plt.ylabel("Stocks price of (High) category")
plt.xticks(fontsize=12); plt.yticks(fontsize=12)
plt.title("Scatterplot of Stock prices", fontsize=22)
plt.show()    


Stocks_data.boxplot(column = "High")
plt.show()

Stocks_data.boxplot(column = "Low")
plt.show()

#The above boxplots are showing that there are no outliers and also showing the lowest value of stock price that is about 380 approx.

sns.catplot(data = Stocks_data, x = 'Low', y = 'High')
plt.show()

sns.FacetGrid(Stocks_data, size=5) \
   .map(plt.scatter, "Low", "High") \
   .add_legend()
plt.show()


# Same thing as catplot

# Viewing all the Integer columns
ints = Stocks_data.select_dtypes(exclude = 'object').columns.to_list()
ints

# Histogram

Stocks_data.hist(color = "violet",
        bins = 30,
        figsize = (15, 10))
plt.show()

#A visual analysis of the histograms presented allows us to make preliminary assumptions about the variability of the source data.

# Plotting a histogram to view how the 'open' and 'close' feature are laid out.

sns.histplot(data=Stocks_data["Open"] , color="red", label="Opening of Stocks Price")
sns.histplot(data=Stocks_data["Close"] , color="blue", label="Closing of Stocks Price")
plt.title("Comparision between opening one & closing one Stock prices")
plt.show()

plt.figure(figsize = (5, 4))
sns.histplot(data = Stocks_data, x = "High", kde = True)
plt.show()

#The histogram showing the increase in the Stock Prices.

plt.figure(figsize = (5, 4))
sns.histplot(data = Stocks_data, x = "Low", kde = True)
plt.show()

#The histogram showing the decrease in the Stock Prices.

sns.histplot(data = Stocks_data["Volume"] , color="green", label="Volume of the Stocks")
plt.title("Volume")
plt.show()

# Heatmap

plt.figure(figsize = (12, 6))
sns.heatmap(Stocks_data.corr(), annot = True, cmap = "coolwarm")
plt.show()

# You will not loose any information if you cut down one part of the heatmap along the diagonal

mask = np.triu(np.ones_like(Stocks_data.corr(), dtype = np.bool))
plt.figure(figsize = (12, 6))
heatmap = sns.heatmap(Stocks_data.corr(), cmap = "coolwarm", annot = True, mask = mask)
heatmap.set_title("Heat Map", fontsize = 20, pad = 10)
plt.show()


correlation = Stocks_data.corr()[['Low']].sort_values(by = 'Low', ascending = False)
print(correlation)

correlation2 = Stocks_data.corr()[['High']].sort_values(by = 'High', ascending = False)
print(correlation2)

#Observation
#The correlation between the columns,

#Here you can see the above cell in descending order the close is about 99 percent correlated and open is also 99 percent correlated and so on in descending order

#Same you can check from heatmap plotted Below------

plt.figure(figsize = (12, 6))
heatmap = sns.heatmap(correlation, cmap = "coolwarm", vmax = 1, annot = True)
heatmap.set_title("Correlation of independent variable", fontsize = 10)
plt.show()

plt.figure(figsize = (12, 6))
heatmap = sns.heatmap(correlation2, cmap = "coolwarm", vmax = 1, annot = True)
heatmap.set_title("Correlation of independent variable", fontsize = 10)
plt.show()

#Insights
#We will select the columns which are positevely correlated in our dataset while selecting negatively correlated data as you can see in the heatmap that High column is highly correlated but Volume is 0.4 percent correlated but negatively correlated. 
#The red column is highly correlated while the blue column is less correlated

Stocks_data['Adj_Close'].plot()
plt.show()

#Now, the basic Cleaning, visualization, EDA and feature Engineering is done , now Building the machine learning models.
#Prediction Part----

# Required libraries for data preprocessing and predictions

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# Setting the target values for prediction 


#Selecting the Features
X = Stocks_data[['Open', 'High', 'Low', 'Volume']]
Y = Stocks_data.Adj_Close

# Dividing the data into training and testing 

X_train, X_test, y_train, y_test = train_test_split(X,Y,
                                                      train_size=0.8, test_size=0.2,
                                                      random_state=0)

X_train.shape ,X_test.shape ,y_train.shape  , y_test.shape # Training and testing data

### Let's apply Machine learning algorithms or Regressors

y_test = np.array(y_test)

#1) LSTM MODEL(LONG SHORT TERM MEMORY MODEL)

# Building the model

regressor = Sequential()

regressor.add(LSTM(units=50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))

regressor.add(Dense(units = 1))

# Fitting the model

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
history = regressor.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 100, batch_size = 32)
history

history_Stocks= pd.DataFrame(history.history)
history_Stocks.loc[0:, ['loss', 'val_loss']].plot()
print(("Minimum Validation Loss: {:0.4f}").format(history_Stocks['val_loss'].min()))

len(Stocks_data)

actual_training_set = Stocks_data.iloc[:, 1:2].values
actual_training_set.ndim

scaler = MinMaxScaler(feature_range = (0,1))
scaled_traning_set = scaler.fit_transform(actual_training_set)
scaled_traning_set

predicted_stock_price = regressor.predict(X_test)

reshape_stock_price = predicted_stock_price.reshape(-1, 1)
reshape_stock_price.shape

predicted_stock_price_Total = scaler.inverse_transform(reshape_stock_price)
predicted_stock_price_Total.shape

plt.plot(actual_training_set, color = 'green', label = 'Actual Stocks Price')
plt.plot(predicted_stock_price_Total, color = 'yellow', label = 'Predicted Stocks Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

predicted_stock_price_Total = predicted_stock_price.reshape(-1)
actual_training_set = actual_training_set.reshape(-1)

actual_training_set.shape

Actual = pd.DataFrame({'Actual':actual_training_set})
Actual

Predicted = pd.DataFrame({'Predicted Monthly Sales': predicted_stock_price_Total})
Predicted

#Now, Machine learning algorithms for prediction---

# Importing required libraries first

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

Dt_model = DecisionTreeRegressor(random_state = 1)
Dt_model.fit(X_train, y_train)

print('[1]Decision Tree Regressor Training Accuracy:', Dt_model.score(X_train, y_train))

dt_preds = Dt_model.predict(X_test)

print("MAE from Approach 2 Decision Tree Regressor:")
mean_absolute_error(y_test, dt_preds)

comparison = pd.DataFrame({'Actual':y_test, 'Predicted': dt_preds})
comparison

plt.plot(dt_preds , color = 'green', label = 'Predicted Stocks Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

plt.plot(y_test, color = 'blue', label = 'Actual Stocks Price')
plt.plot(dt_preds , color = 'green', label = 'Predicted Stocks Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()

#Observation
#The calculated accuracy of the Decision Tree model is 1.0.

#The calculated error of the Decision Tree model is 4.26059578217822.

# Importing required libraries first

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Method for comparing different approaches
model = RandomForestRegressor(n_estimators=100, random_state=0) # 100 are the forests random forests
model.fit(X_train, y_train)
rf_preds = model.predict(X_test)

print('[2]Random Forest Classifier Training Accuracy:', model.score(X_train, y_train))

print("MAE from Approach 1 Random Forest Regressor:")
print(mean_absolute_error(rf_preds, y_test))

comparison = pd.DataFrame({'Actual':y_test, 'Predicted': rf_preds})
comparison

actual_training_values= np.array(comparison)

plt.plot(y_test, color = 'blue', label = 'Actual Stocks Price')
plt.plot(rf_preds, color = 'green', label = 'Predicted Stocks Price')
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()


#Result**
#Hence, we checked and found that using machine Learning models we can automate the investment predictions process.

