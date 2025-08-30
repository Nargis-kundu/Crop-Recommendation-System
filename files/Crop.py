import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import pickle

# Load the dataset
crop = pd.read_csv('Crop_recommendation.csv')

# Display first few rows
print(crop.head())

# Check dataset shape and info
print("Dataset Shape:", crop.shape)
crop.info()

# Checking for missing values
print("Missing Values:\n", crop.isnull().sum())

# Checking for duplicate values
print("Duplicate Values:", crop.duplicated().sum())

# Summary statistics
print("Dataset Summary:\n", crop.describe())

# Check correlation between numeric columns
numeric_crop = crop.select_dtypes(include=[np.number])
corr = numeric_crop.corr()
print("Feature Correlation:\n", corr)

# Optional: Heatmap of correlation
# sns.heatmap(corr, annot=True, cmap='coolwarm')
# plt.show()

# Encode 'label' using LabelEncoder
le = LabelEncoder()
crop['crop_num'] = le.fit_transform(crop['label'])

# Drop original label
crop.drop(columns=['label'], inplace=True)

# Splitting data into features (X) and target (y)
X = crop.drop('crop_num', axis=1)
y = crop['crop_num']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest Classifier
rfc = RandomForestClassifier()
rfc.fit(X_train, y_train)

# Predictions
y_pred = rfc.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy:", accuracy)

# Function to recommend a crop
def recommendation(N, P, K, temperature, humidity, ph, rainfall):
    features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    features = scaler.transform(features)
    prediction = rfc.predict(features)
    return prediction[0]  # Return predicted class number

# Reverse mapping from encoded number to crop name
crop_dict_reverse = {i: label for i, label in enumerate(le.classes_)}

# User input
try:
    N = int(input("Enter Nitrogen (N) value(0 <= N <= 140): "))
    P = int(input("Enter Phosphorus (P) value(5 <= P <= 145): "))
    K = int(input("Enter Potassium (K) value(5 <= K <= 205): "))
    temperature = float(input("Enter Temperature (°C)(8 <= temperature <= 55): "))
    humidity = float(input("Enter Humidity (%)(10 <= humidity <= 100 ): "))
    ph = float(input("Enter pH level(2.5 <= ph <= 14): "))
    rainfall = float(input("Enter Rainfall (mm)(20 <= rainfall <= 300): "))
except ValueError:
    print("Invalid input. Please enter numerical values only.")
    exit()

# Optional: Input validation
if not (0 <= N <= 140 and 5 <= P <= 145 and 5 <= K <= 205 and
        8 <= temperature <= 56 and 10 <= humidity <= 100 and
        2.5 <= ph <= 14 and 20 <= rainfall <= 300):
    print("One or more inputs are out of the valid range.")
    exit()

# Predict crop
predicted_crop_num = recommendation(N, P, K, temperature, humidity, ph, rainfall)

# Output result
predicted_crop = crop_dict_reverse.get(predicted_crop_num, "Unknown")
print(f"Recommended Crop: {predicted_crop}")

# Save model and scaler
with open('model.pkl', 'wb') as f:
    pickle.dump(scaler, f)
    pickle.dump(rfc, f)

# Save label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

