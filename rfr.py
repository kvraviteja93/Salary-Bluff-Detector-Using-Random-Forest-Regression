#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing The Dataset
dataset=pd.read_csv("Position_Salaries.csv")

#Splitting Independent and Dependent Variables
X=dataset.iloc[:,1:2].values
y=dataset.iloc[:,-1].values

#Fitting Random Forest Regression To The Dataset
from sklearn.ensemble import RandomForestRegressor
regressor=RandomForestRegressor(n_estimators=300,random_state=0)
regressor.fit(X,y)

#Predicting The Results 
y_pred=regressor.predict(X)

#Predicting a New Observation
y_pred=regressor.predict(6.5)

#Visualizing The Results
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = 'red')
plt.plot(X_grid, regressor.predict(X_grid), color = 'blue')
plt.title('Truth or Bluff (Random Forest Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
