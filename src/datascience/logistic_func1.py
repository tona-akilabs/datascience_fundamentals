import pandas as pd
from sklearn.linear_model import LogisticRegression


# Load the data
df = pd.read_csv('https://bit.ly/33ebs2R', delimiter=",")
print(df.shape) # (21, 2) => (rows, columns)
print(df.head()) # top 5 rows
# Extract input variables (all rows, all columns but last column)
X = df.values[:, :-1]
# Extract output column (all rows, last column)
Y = df.values[:, -1]
# Perform logistic regression
# Turn off penalty
model = LogisticRegression(penalty=None)
model.fit(X, Y)
# print beta1
print(model.coef_.flatten()) # 0.69267212
# print beta0
print(model.intercept_.flatten()) # -3.17576395
