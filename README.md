



### Name: Koduru Sanath Kumar Reddy
### Register no:212221240024
### Date:

# Ex.No: 07                                       AUTO REGRESSIVE MODEL



### AIM:
To Implementat an Auto Regressive Model using Python
### ALGORITHM:
1. Import necessary libraries
2. Read the CSV file into a DataFrame
3. Perform Augmented Dickey-Fuller test
4. Split the data into training and testing sets.Fit an AutoRegressive (AR) model with 13 lags
5. Plot Partial Autocorrelation Function (PACF) and Autocorrelation Function (ACF)
6. Make predictions using the AR model.Compare the predictions with the test data
7. Calculate Mean Squared Error (MSE).Plot the test data and predictions.
### PROGRAM
~~~
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error


data = pd.read_csv('Electric_Production.csv', parse_dates=['DATE'], index_col='DATE')


plt.figure(figsize=(10, 6))
plt.plot(data, label='Electric Production')
plt.title('Given Electric Production Data')
plt.xlabel('Date')
plt.ylabel('Production')
plt.legend()
plt.show()


adf_result = adfuller(data)
print(f'ADF Statistic: {adf_result[0]}')
print(f'p-value: {adf_result[1]}')


train_size = int(len(data) * 0.8)
train, test = data.iloc[:train_size], data.iloc[train_size:]


lag_order = 13
model = AutoReg(train, lags=lag_order)
model_fitted = model.fit()


plt.figure(figsize=(12, 8))
plt.subplot(211)
plot_acf(data, lags=40, ax=plt.gca())
plt.title('Autocorrelation Function (ACF)')


plt.subplot(212)
plot_pacf(data, lags=40, ax=plt.gca())
plt.title('Partial Autocorrelation Function (PACF)')
plt.tight_layout()
plt.show()

predictions = model_fitted.predict(start=len(train), end=len(train) + len(test) - 1, dynamic=False)
plt.figure(figsize=(10, 6))
plt.plot(test.index, test, label='Actual Test Data', color='blue')
plt.plot(test.index, predictions, label='Predicted Data', color='red')
plt.title('Test Data vs Predictions')
plt.xlabel('Date')
plt.ylabel('Electric Production')
plt.legend()
plt.show()
mse = mean_squared_error(test, predictions)
print(f'Mean Squared Error: {mse}')
~~~
### OUTPUT:

GIVEN DATA


<img width="680" alt="image" src="https://github.com/user-attachments/assets/aab51f8d-becc-4cf5-bc7e-07586f96524d">

PACF - ACF

<img width="971" alt="image" src="https://github.com/user-attachments/assets/bf13d05a-e7c4-4547-8475-469d2a7b66de">

<img width="971" alt="image" src="https://github.com/user-attachments/assets/49372e7f-7f0c-4a39-a102-dd4f8dedea40">


PREDICTION

<img width="738" alt="image" src="https://github.com/user-attachments/assets/53d3ae1d-b368-466b-a324-2dbda03d42d6">



FINIAL PREDICTION

<img width="738" alt="image" src="https://github.com/user-attachments/assets/8687a879-32a5-4020-a2ee-32ff41e51829">




### RESULT:
Thus we have successfully implemented the auto regression function using python.
