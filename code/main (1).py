import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 1. DATA REPRESENTATION 
# Simulating industrial sensor data: Temperature, Vibration, and Hours Operated
np.random.seed(42)
n_samples = 200
vibration = np.random.normal(5, 1, n_samples)  # Common Distribution 
temperature = np.random.normal(70, 5, n_samples) 
hours_run = np.random.uniform(100, 2000, n_samples)

# Target: Remaining Useful Life (RUL) in hours
# RUL decreases as vibration and temperature increase
rul = 3000 - (1.2 * hours_run) - (50 * vibration) - (5 * temperature) + np.random.normal(0, 50, n_samples)

# Create DataFrame
df = pd.DataFrame({
    'Vibration': vibration,
    'Temperature': temperature,
    'Hours_Run': hours_run,
    'RUL': rul
})

# 2. FEATURE LEARNING & VALIDATION SETS 
X = df[['Vibration', 'Temperature', 'Hours_Run']]
y = df['RUL']

# Splitting into Training and Validation sets to analyze Bias and Variance 
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. LINEAR REGRESSION MODEL 
# Applying the regression algorithm 
model = LinearRegression()
model.fit(X_train, y_train)

print("\n--- Make a New Prediction ---")

try:
    # Get user input for new data points
    user_vibration = float(input("Enter current Vibration (e.g., 5.0): "))
    user_temperature = float(input("Enter current Temperature (e.g., 70.0): "))
    user_hours_run = float(input("Enter current Hours Run (e.g., 1500.0): "))

    # Create a DataFrame for the new input
    new_data = pd.DataFrame([{
        'Vibration': user_vibration,
        'Temperature': user_temperature,
        'Hours_Run': user_hours_run
    }])

    # Predict RUL for the new data
    new_prediction = model.predict(new_data)

    print(f"Predicted Remaining Useful Life (RUL) for your input: {new_prediction[0]:.2f} hours")

except ValueError:
    print("Invalid input. Please enter numerical values for Vibration, Temperature, and Hours Run.")
except Exception as e:
    print(f"An error occurred: {e}")