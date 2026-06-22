import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Dataset
data = {
    "Sales": [12000,15000,18000,17000,21000,25000,27000,29000,31000,35000,37000,42000],
    "Customers": [120,150,180,170,210,250,270,290,310,350,370,420],
    "Advertising": [2000,2500,3000,2800,3500,4000,4500,5000,5500,6000,6500,7000]
}

df = pd.DataFrame(data)

print("Dataset")
print(df)

# Statistical Summary
print("\nSummary")
print(df.describe())

# Correlation
print("\nCorrelation")
print(df.corr())

# Features and Target
X = df[["Customers","Advertising"]]
y = df["Sales"]

# Train-Test Split
X_train,X_test,y_train,y_test = train_test_split(
    X,y,test_size=0.2,random_state=42
)

# Linear Regression
model = LinearRegression()
model.fit(X_train,y_train)

# Prediction
pred = model.predict(X_test)

# Error
mae = mean_absolute_error(y_test,pred)

print("\nMean Absolute Error:",mae)

# Predict Future Sales
future = model.predict([[450,7500]])

print("\nPredicted Future Sales:",future)

# Sales Trend
plt.figure(figsize=(8,5))
plt.plot(df["Sales"], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month Index")
plt.ylabel("Sales")
plt.show()

# Customers vs Sales
plt.figure(figsize=(8,5))
plt.scatter(df["Customers"],df["Sales"])
plt.title("Customers vs Sales")
plt.xlabel("Customers")
plt.ylabel("Sales")
plt.show()