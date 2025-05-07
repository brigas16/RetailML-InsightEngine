import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Set style
sns.set(style="whitegrid")

# Load and preprocess data
df = pd.read_csv('your_data.csv')
df.columns = ['Date', 'Time', 'Item', 'Price', 'Quantity', 'Total',
              'Customer ID', 'Payment Method', 'Employee ID',
              'Customer Satisfaction', 'Weather', 'Special Offers']
df.dropna(inplace=True)
df['DateTime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'])
df.drop(columns=['Date', 'Time', 'Total'], inplace=True)

# Features
cat_features = ['Payment Method', 'Special Offers', 'Weather']
num_features = ['Price', 'Quantity', 'Customer ID', 'Employee ID']
target = 'Customer Satisfaction'

# Preprocessing
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features),
    ('num', StandardScaler(), num_features)
])

# Prepare features and labels
X = preprocessor.fit_transform(df[cat_features + num_features])
y = df[target].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Neural Network with TensorFlow ---
input_dim = X_train.shape[1]
model = Sequential([
    Dense(128, activation='relu', input_shape=(input_dim,)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(1)  # Regression output
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Fit model
history = model.fit(X_train, y_train, epochs=100, batch_size=32,
                    validation_split=0.2, callbacks=[early_stop], verbose=0)

# Evaluate
y_pred_nn = model.predict(X_test).flatten()
rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred_nn))
r2_nn = r2_score(y_test, y_pred_nn)
print(f"TensorFlow NN - RMSE: {rmse_nn:.2f}, R²: {r2_nn:.2f}")

# Plot loss curve
plt.figure(figsize=(8, 4))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.legend()
plt.show()

# --- Visuals (same as before) ---
# 1. Distribution
plt.figure(figsize=(8, 4))
sns.histplot(df[target], bins=5, kde=True)
plt.title("Distribution of Customer Satisfaction")
plt.xlabel("Satisfaction Score (1-5)")
plt.ylabel("Frequency")
plt.show()

# 2. Correlation Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(df[['Price', 'Quantity', 'Customer ID', 'Employee ID', target]].corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlations")
plt.show()

# 3. Actual vs Predicted
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred_nn, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.title("Actual vs Predicted - TensorFlow Model")
plt.xlabel("Actual")
plt.ylabel("Predicted")
plt.show()

# 4. Payment Method
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='Payment Method')
plt.title("Payment Methods Used")
plt.xlabel("Method")
plt.ylabel("Count")
plt.show()

# 5. Save metrics
results = pd.DataFrame({
    "Metric": ["RMSE", "R² Score"],
    "RandomForest": [np.nan, np.nan],  # fill later if needed
    "TensorFlow NN": [rmse_nn, r2_nn]
})
print("\nModel Performance Table:")
print(results.to_string(index=False))
