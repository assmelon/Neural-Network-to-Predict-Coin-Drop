import numpy as np
import tensorflow
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras import models, layers

df = pd.read_csv('Numerical_coin_data.csv') # Uploading the dataset

features = df[['Landed Coin', 'Drop Orientation', 'Distance to origin']]  # The inputs being chosen
label1 = df[['Landed Coin']]
label2 = df[['Distance to origin']]  # And the two outputs we have chosen

print(features)
print(label1)
print(label2)

# Dividing the dataset into training and test datasets
features_train, features_test, label1_train, label1_test, label2_train, label2_test = train_test_split(features, label1, label2, test_size=0.2, random_state=42)

# Creating the model
model = models.Sequential([
    layers.Dense(64, activation='relu', input_shape=(features_train.shape[1],)),
    layers.Dense(32, activation='relu'),
    layers.Dense(2)  # Output layer with 2 neurons for the two target variables
])

# Compiling and choosing the metrics we want to see
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Fitting the model and choosing epochs
model.fit(features_train, [label1_train, label2_train], epochs=10, batch_size=32)

# Evaluating the accuracy
test_loss, test_acc = model.evaluate(features_test, [label1_test, label2_test])
print('Test Accuracy:', {test_acc})

# Printing the predictions, mse and mae
predictions = model.predict(features_test)
print("Predictions:\n", predictions[:5])

evaluation = model.evaluate(features_test, label1_test)
print("\nEvaluation Metrics:")
print(f"Mean Squared Error: {evaluation[0]}")
print(f"Mean Absolute Error: {evaluation[1]}")
