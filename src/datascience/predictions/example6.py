from sklearn.linear_model import LogisticRegression

# Features: email length
X = [[100], [200], [300], [400]]
y = [0, 0, 1, 1]  # 0 = Not Spam, 1 = Spam

model = LogisticRegression()
model.fit(X, y)

# Predict new email
result = model.predict([[250]])
print("Spam?" , result[0])