import numpy as np
from sklearn.linear_model import LinearRegression

# Data: house size (m²) → price ($)
X = np.array([50, 60, 80, 100, 120]).reshape(-1, 1)
y = np.array([100000, 120000, 160000, 200000, 240000])

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict price for 90 m² house
prediction = model.predict([[90]])

print("Predicted price:", prediction[0]) # 180000.0