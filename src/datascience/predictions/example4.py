import numpy as np
from sklearn.linear_model import LinearRegression

# Time (days)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([100, 120, 130, 150, 170])  # sales

model = LinearRegression()
model.fit(X, y)

# Predict day 6
future_sales = model.predict([[6]])
print("Predicted sales:", future_sales[0])