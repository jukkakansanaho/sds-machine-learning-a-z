# Simple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

# Training the Simple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

# BONUS:
# Question 1: How do I use my simple linear regression model to make a single prediction, for example to predict the salary of an employee with 12 years of experience?
# Question 2: How do I get the final regression equation y = b0 + b1 x with the final values of the coefficients b0 and b1?
#
# These two questions are answered in detail with the explanation and code at the end of this Bonus Colab file:
# https://colab.research.google.com/drive/1934DQETINwi7yt-wE3nDd4CxH8ZZNJlE?usp=sharing

# Answer-1:
print(regressor.predict([[12]]))

# Answer-2:
print(regressor.coef_)
print(regressor.intercept_)
