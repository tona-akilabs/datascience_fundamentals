import pandas as pd
import numpy as np

from sklearn.linear_model import LinearRegression

data = pd.DataFrame([
    {"size_m2": 85, "bedrooms": 3, "age_years": 4, "city": "B"},
    {"size_m2": 60, "bedrooms": 2, "age_years": 9, "city": "C"},
    {"size_m2": 75, "bedrooms": 2, "age_years": 3, "city": "A"},
])

# One-hot encode city
data_encoded = pd.get_dummies(data, columns=["city"])
print(data_encoded)

data.head()

X = data_encoded.drop(columns=["size_m2"])
y = data_encoded["size_m2"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Predict for:
new_data = pd.DataFrame([{
    "bedrooms": 2,
    "age_years": 2,
    "city_A": True,
    "city_B": False,
    "city_C": False
}])

prediction = model.predict(new_data)
print(prediction)
print("Predicted size (m²):", np.round(prediction[0], 2))