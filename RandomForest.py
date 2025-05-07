import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set(style="whitegrid")
# Load your CSV data (replace 'your_data.csv' with the actual file)
df = pd.read_csv('your_data.csv')

# Example: If your CSV has columns like Date, Time, etc.
# Ensure column names match exactly
df.columns = ['Date', 'Time', 'Item', 'Price', 'Quantity', 'Total', 
              'Customer ID', 'Payment Method', 'Employee ID', 'Customer Satisfaction', 'Weather', 'Special Offers']
# Handle missing values (if any)
df.dropna(inplace=True)

# Convert to datetime
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
# Drop redundant columns (e.g., Total if derived from Price and Quantity)
df.drop(columns=['Date', 'Time', 'Total'], inplace=True)

# Categorical features to encode
cat_features = ['Payment Method', 'Special Offers', 'Weather']
num_features = ['Price', 'Quantity', 'Customer ID', 'Employee ID']

# One-hot encode categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
        ('num', StandardScaler(), num_features)
    ])

# Target variable
y = df['Customer Satisfaction'].values
X = preprocessor.fit_transform(df[cat_features + num_features])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use RandomForestRegressor (or any model)
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")
# 1. Distribution of Customer Satisfaction (Histogram)
plt.figure(figsize=(8, 4))
sns.histplot(df['Customer Satisfaction'], bins=5, kde=True)
plt.title("Distribution of Customer Satisfaction")
plt.xlabel("Satisfaction Score (1-5)")
plt.ylabel("Frequency")
plt.show()

# 2. Correlation Matrix
numeric_df = df[['Price', 'Quantity', 'Customer ID', 'Employee ID', 'Customer Satisfaction']]
plt.figure(figsize=(8, 6))
sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlations")
plt.show()

# 3. Feature Importance
importances = model.named_steps['regressor'].feature_importances_
features = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features)
num_features_list = num_features

all_features = features + num_features_list
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=all_features)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.ylabel("Features")
plt.show()

# 4. Actual vs Predicted Values
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted Customer Satisfaction")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# 5. Payment Method Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Payment Method')
plt.title("Payment Methods Used")
plt.xlabel("Method")
plt.ylabel("Count")
plt.show()
# Save metrics to a table
results = pd.DataFrame({
    "Metric": ["RMSE", "R² Score"],
    "Value": [rmse, r2]
})
print("\nModel Performance Table:")
print(results.to_string(index=False))
