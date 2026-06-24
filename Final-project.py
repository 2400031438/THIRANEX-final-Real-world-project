# ==========================================
# Retail Sales Prediction Using Linear Regression
# Complete Project Code (Without CSV File)
# ==========================================

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# STEP 1: Create Dataset
# ==========================================

data = {
    "Month": ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"],
    
    "Sales": [12000, 15000, 18000, 17000, 21000, 25000,
              27000, 29000, 31000, 35000, 37000, 42000],
    
    "Customers": [120, 140, 160, 150, 180, 220,
                  240, 260, 280, 310, 330, 370]
}

df = pd.DataFrame(data)

# ==========================================
# STEP 2: Display Dataset
# ==========================================

print("Retail Sales Dataset")
print(df)

# ==========================================
# STEP 3: Data Analysis
# ==========================================

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ==========================================
# STEP 4: Sales Trend Visualization
# ==========================================

plt.figure(figsize=(8,5))
plt.plot(df["Month"], df["Sales"], marker='o')
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales (₹)")
plt.grid(True)
plt.show()

# ==========================================
# STEP 5: Customer Growth Visualization
# ==========================================

plt.figure(figsize=(8,5))
plt.bar(df["Month"], df["Customers"])
plt.title("Monthly Customer Count")
plt.xlabel("Month")
plt.ylabel("Number of Customers")
plt.show()

# ==========================================
# STEP 6: Relationship Between Customers & Sales
# ==========================================

plt.figure(figsize=(8,5))
plt.scatter(df["Customers"], df["Sales"])
plt.title("Customers vs Sales")
plt.xlabel("Customers")
plt.ylabel("Sales (₹)")
plt.grid(True)
plt.show()

# ==========================================
# STEP 7: Prepare Data for Machine Learning
# ==========================================

X = df[["Customers"]]
y = df["Sales"]

# Split dataset into training and testing data

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

# ==========================================
# STEP 8: Train Linear Regression Model
# ==========================================

model = LinearRegression()

model.fit(X_train, y_train)

print("\nModel Training Completed")

# ==========================================
# STEP 9: Predict Test Data
# ==========================================

predictions = model.predict(X_test)

print("\nActual Sales")
print(y_test.values)

print("\nPredicted Sales")
print(predictions)

# ==========================================
# STEP 10: Evaluate Model
# ==========================================

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\nModel Evaluation")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 4))

# ==========================================
# STEP 11: Compare Actual vs Predicted
# ==========================================

comparison = pd.DataFrame({
    "Actual Sales": y_test.values,
    "Predicted Sales": predictions
})

print("\nComparison Table")
print(comparison)

# ==========================================
# STEP 12: Visualization of Predictions
# ==========================================

plt.figure(figsize=(8,5))
plt.scatter(X_test, y_test, label="Actual Data")
plt.plot(X_test, predictions, linewidth=2,
         label="Regression Line")

plt.title("Linear Regression Prediction")
plt.xlabel("Customers")
plt.ylabel("Sales (₹)")
plt.legend()
plt.grid(True)
plt.show()

# ==========================================
# STEP 13: Future Sales Prediction
# ==========================================

future_customers = [[400]]

future_sales = model.predict(future_customers)

print("\nFuture Prediction")
print("Customers:", future_customers[0][0])
print("Predicted Sales: ₹", round(future_sales[0], 2))

# ==========================================
# STEP 14: Conclusion
# ==========================================

print("\nPROJECT CONCLUSION")
print("1. Sales increase as customer count increases.")
print("2. Strong positive relationship exists between customers and sales.")
print("3. Linear Regression predicts future sales effectively.")
print("4. Businesses can use this model for sales forecasting.")