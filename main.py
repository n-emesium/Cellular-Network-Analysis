import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Step 1: Data Preparation
# Load the data
df = pd.read_csv('signal_metrics.csv')

# Set target variable (Y) and input features (X)
Y = df['Data Throughput (Mbps)']
X = df[['Signal Strength (dBm)', 'Latency (ms)', 'Network Type', 
        'BB60C Measurement (dBm)', 'srsRAN Measurement (dBm)', 
        'BladeRFxA9 Measurement (dBm)']]

# One-Hot Encode the 'Network Type' categorical variable
column_transformer = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(), ['Network Type'])
    ],
    remainder='passthrough'
)

X = column_transformer.fit_transform(X)

# Convert to DataFrame to allow for filling missing values
X = pd.DataFrame(X)

# Data Cleaning: Check for missing values and handle them
# Fill missing values in numerical columns with mean
numerical_columns = X.columns[1:]  # skip the first column since it's categorical
X[numerical_columns] = X[numerical_columns].apply(lambda col: col.fillna(col.mean()), axis=0)

# Split the dataset into training and test sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Step 2: Feature Engineering
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 3: Model Development
# Model Architecture
model = Sequential()

# Input layer and hidden layers
model.add(Dense(128, input_shape=(X_train_scaled.shape[1],), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))  # Regularization
model.add(Dense(32, activation='relu'))

# Output layer
model.add(Dense(1, activation='linear'))  # Output layer for regression

# Model Compilation
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Model Training
# Training with EarlyStopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train_scaled, Y_train, 
                    validation_data=(X_test_scaled, Y_test),
                    epochs=100, 
                    batch_size=32, 
                    callbacks=[early_stopping],
                    verbose=1)

# Step 5: Model Evaluation and Results
# Evaluate the model on the test set
test_loss = model.evaluate(X_test_scaled, Y_test, verbose=0)
print(f'Test Loss (MSE): {test_loss}')

# Predict on test data
Y_pred = model.predict(X_test_scaled)

# Visualize predictions vs actual values
plt.scatter(Y_test, Y_pred)
plt.xlabel('Actual Data Throughput (Mbps)')
plt.ylabel('Predicted Data Throughput (Mbps)')
plt.title('Actual vs Predicted Data Throughput')
plt.show()
