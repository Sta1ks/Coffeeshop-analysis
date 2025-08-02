#In this file, I would create machine learning models for Pastry, Biscotti and Scone
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def get_data(name: str): 
   return pd.read_csv("model and products/"+name)

def get_target_and_features(data): #day feature might be redundant because there are already features like
    return data["revenue_per_day"].values, data.iloc[:, 2:].values #day_of_week, month, so multicollinearity.

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def regression(X_train, X_test, y_train, y_test):
    reg = LinearRegression()
    reg.fit(X_train, y_train)
    y_pred = reg.predict(X_test)
    for actual, pred in zip(y_test, y_pred):  
     print(f"Actual: {actual:.2f}  |  Predicted: {pred:.2f}")

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\nRMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R² Score: {r2:.2f}")


Pastry = get_data("Pastry.csv")
Biscotti = get_data("Biscotti.csv")
Scone = get_data("Scone.csv")

#I will create several models to predict expected revenue for each of the bakery category.
#From the predicted revenue, it would be also clear how much quantity of each product is
#it better to make to optimize production. 

y_Pastry, X_pastry = get_target_and_features(Pastry)
y_Biscotti, X_Biscotti = get_target_and_features(Biscotti)
y_Scone, X_Scone = get_target_and_features(Scone)

X_train_Pastry, X_test_Pastry, y_train_Pastry, y_test_Pastry = split(X_pastry, y_Pastry)
X_train_Biscotti, X_test_Biscotti, y_train_Biscotti, y_test_Biscotti = split(X_Biscotti, y_Biscotti)
X_train_Scone, X_test_Scone, y_train_Sconey, y_test_Scone = split(X_Scone, y_Scone)

regression(X_train_Pastry, X_test_Pastry, y_train_Pastry, y_test_Pastry)
regression(X_train_Scone, X_test_Scone, y_train_Sconey, y_test_Scone)
regression(X_train_Biscotti, X_test_Biscotti, y_train_Biscotti, y_test_Biscotti)


