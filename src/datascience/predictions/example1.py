import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# Load built-in dataset
df = sns.load_dataset('mpg')
print(df.head())
print(df.info())
print(df.shape)

# Check for missing values
print(df.isnull().sum())

# Drop rows with missing values (horsepower often has a few)
df = df.dropna()

# Visualize relationship
sns.regplot(data=df, x='weight', y='mpg')
plt.title("Weight vs. Fuel Efficiency")
plt.show()

# Select features and target
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'origin']
X = df[features]
y = df['mpg']

# Convert categorical 'origin' into dummy variables (0 and 1s)
X = pd.get_dummies(X, columns=['origin'], drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print(f"R2 Score: {r2_score(y_test, predictions):.2f}")
print(f"Average Error: {mean_absolute_error(y_test, predictions):.2f} MPG")