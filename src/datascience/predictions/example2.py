import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = pd.DataFrame({
    "size_m2":   [50, 60, 70, 80, 90, 100, 110, 120, 55, 95, 130, 75],
    "bedrooms":  [1,  2,  2,  3,  3,  3,   4,   4,  2,  3,  4,  2],
    "age_years": [10, 8,  6,  5,  3,  2,   1,   1,  9,  4,  1,  7],
    "city":      ["A","A","A","B","B","B","B","C","C","C","B","A"],
    "price":     [55, 65, 75, 92, 105, 112, 125, 120, 70, 110, 140, 78]  # target
})

data.head()

X = data.drop(columns=["price"])
y = data["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

numeric_features = ["size_m2", "bedrooms", "age_years"]
categorical_features = ["city"]

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ]
)
model = Pipeline(steps=[
    ("preprocess", preprocess),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
# print(X_test)
# print(y_test)
# print(y_train)
# print(X_train)
results = pd.DataFrame({
    "actual": y_test.values,
    "predicted": np.round(y_pred, 2)
})

print(results)

# mae = mean_absolute_error(y_test, y_pred)
# rmse = mean_squared_error(y_test, y_pred) ** 0.5
# r2 = r2_score(y_test, y_pred)
#
# print("MAE :", round(mae, 2))
# print("RMSE:", round(rmse, 2))
# print("R2  :", round(r2, 2))

# plt.scatter(y_test, y_pred)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted")
# plt.show()
