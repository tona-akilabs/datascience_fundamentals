from sklearn.tree import DecisionTreeClassifier

# Features: [study_hours, attendance]
X = [
    [2, 60],
    [4, 70],
    [6, 80],
    [8, 90]
]
y = ["Fail", "Fail", "Pass", "Pass"]

model = DecisionTreeClassifier()
model.fit(X, y)

# Predict new student
result = model.predict([[5, 75]])

# Predict new student 2
#result = model.predict([[7, 75]])
print("Prediction:", result[0])