#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import RFE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error

#Read Data
data = pd.read_csv('ames.csv')

#Examine Data
print(data.head(5))

#Fill in NaN Values
data = data.fillna(0)

#EDA - Plot Sale Price
sns.distplot(data['SalePrice'], bins=50, hist=True, kde=False)
plt.xlabel('Sale Price ($)')
plt.ylabel('Count')
plt.show()

#Log Sale Price to Improve Modelling
SalePrice = np.log(data['SalePrice'])

#Convert Categorical Data to Numerical
data = pd.get_dummies(data)

#Initiate Model
model = LassoCV()

#Select Features Using RFE
rfe = RFE(model, 3)
rfe = rfe.fit(data, SalePrice)
data = rfe.transform(data)

#Split Data into Train and Test Sets
X_train, X_test, y_train, y_test = train_test_split(data, SalePrice, test_size=0.5, random_state=42, shuffle=True)

#Fit Model to Training Data
model.fit(X_train, y_train)

#Use Model to Predict on Testing Data
y_pred = model.predict(X_test)

#Compute MSE
mse = mean_squared_error(y_test, y_pred)
print('MSE = ', mse)

#Convert Back into ($) Value
y_test = np.exp(y_test)
y_pred = np.exp(y_pred)

#Compute Error between Test and Prediction Value
error = [y_test - y_pred]

#Plot Error
sns.distplot(error, hist=True)
plt.xlabel('Difference Between Prediction and Test Value ($)')
plt.show()

#Calculate the Average Variance in SalePrice
print('The average error in the predicted value is ± $',np.mean(error))

