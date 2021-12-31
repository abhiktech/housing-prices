import pandas as pd # data analysis
import numpy as np # numerical computations
import matplotlib.pyplot as plt # data visualization
# %matplotlib inline
import seaborn as sns # makes data visualization prettier
from sklearn.model_selection import train_test_split # makes it easy to split data into training and test data sets
from sklearn.linear_model import LinearRegression # makes it very easy to build and test linear regression models
from sklearn import metrics # to test model performance

raw_data = pd.read_csv('Housing_Data.csv') # importing the data set

x = raw_data[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms', 'Area Population']]
y = raw_data['Price']
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3)

model = LinearRegression()
model.fit(x_train, y_train) # training the data set

predictions = model.predict(x_test) # making predictions for the test data set
plt.scatter(y_test, predictions) # comparing the predictions with the actual y values in the test data set
plt.show()
plt.hist(y_test - predictions) # computing residuals
plt.show()

print(metrics.mean_absolute_error(y_test, predictions))

print(metrics.mean_squared_error(y_test, predictions))

print(np.sqrt(metrics.mean_squared_error(y_test, predictions)))