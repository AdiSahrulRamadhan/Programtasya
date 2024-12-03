import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
import joblib  # Untuk menyimpan dan memuat model

# Load dataset
health = pd.read_csv('mental_health_wearable_data.csv')  # Ganti dengan nama file Anda

# Separate features (X) and target (y)
X = health.drop(columns=['Mental_Health_Condition'])  # Pastikan 'Mental_Health_Condition' adalah nama kolom label
y = health['Mental_Health_Condition']

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Memisahkan dataset menjadi data latih dan data uji
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=123)

# Build Naive Bayes Gaussian model
gnb_model = GaussianNB()

# Train the model using the training data
gnb_model.fit(X_train, y_train)

# Save the trained model and scaler to files
joblib.dump(gnb_model, 'mental_health_model.pkl')  # Menyimpan model
joblib.dump(scaler, 'scaler.pkl')  # Menyimpan scaler untuk transformasi data baru

print("Model dan scaler telah disimpan!")
