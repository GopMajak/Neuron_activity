# main.py

from Load_neuron_data import load_mat_file, preprocess_data
from Transformer_Model import build_transformer_model
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler

# File path to the MATLAB .mat file
file_path = r'P:\\Projects\\Predicting_Neuron_Activity\\spk_bin10_KA4_2014_05_07.mat'

# Load and preprocess data
mat_data = load_mat_file(file_path)
seq_length = 100
future_steps = 50  # Number of future steps you want to predict
X_train, X_test, y_train, y_test = preprocess_data(mat_data, seq_length, future_steps)

# Scale input data (X_train and X_test)
scaler_X = MinMaxScaler()
X_train = scaler_X.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_test = scaler_X.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Reshape y_train to 2D (n_samples * future_steps, n_features) for scaling
n_samples, future_steps, n_features = y_train.shape
y_train_reshaped = y_train.reshape(-1, n_features)

# Apply MinMaxScaler to y_train
scaler_y = MinMaxScaler()
y_train_scaled = scaler_y.fit_transform(y_train_reshaped)

# Reshape y_train back to 3D (n_samples, future_steps, n_features)
y_train = y_train_scaled.reshape(n_samples, future_steps, n_features)

# Reshape y_test and apply the same scaling
y_test_reshaped = y_test.reshape(-1, n_features)
y_test_scaled = scaler_y.transform(y_test_reshaped)
y_test = y_test_scaled.reshape(y_test.shape[0], future_steps, n_features)

# Define input shape for the Transformer model (seq_length, num_features)
input_shape = (X_train.shape[1], X_train.shape[2])  # (seq_length, num_features)

# Build and compile the Transformer model
model = build_transformer_model(input_shape, future_steps)

# Explicitly build the model before calling summary
model.build(input_shape=(None,) + input_shape)

# Print model summary
model.summary()

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',   # Metric to monitor (e.g., 'val_loss' or 'val_mae')
    patience=5,           # Number of epochs to wait for improvement before stopping
    restore_best_weights=True  # Restores model weights from the epoch with the best metric value
)

# Define ReduceLROnPlateau callback to reduce learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,   # Reduce learning rate by 20%
    patience=3,   # Wait for 3 epochs of no improvement
    min_lr=1e-7   # Minimum learning rate
)

# Train the Transformer model
history = model.fit(
    X_train, y_train,
    epochs=1,  # Adjust epochs based on your data size
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr]  # Include the EarlyStopping callback
)

# Evaluate the model
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")

def plot_forecasting(y_true, y_pred, title='Time Series Forecasting', num_points=100):
    num_neurons = y_true.shape[2]  # For multiple neurons
    num_steps = y_true.shape[1]  # For future steps

    for neuron in range(num_neurons):
        plt.figure(figsize=(14, 8))
        for step in range(num_steps):  # Plot each future step
            plt.plot(y_true[:num_points, step, neuron], label=f'Actual Step {step + 1}', color='black', linestyle='-')
            plt.plot(y_pred[:num_points, step, neuron], label=f'Predicted Step {step + 1}', color='orange', linestyle='--')
        
        plt.title(f"{title} - Neuron {neuron + 1}")
        plt.xlabel('Time Steps')
        plt.ylabel('Neuron Activity')
        plt.legend()
        plt.grid(True)
        plt.show()


# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform predictions and true values for plotting
y_test_inv = scaler_y.inverse_transform(y_test.reshape(-1, n_features))  # Flatten for inverse transform
predictions_inv = scaler_y.inverse_transform(predictions.reshape(-1, n_features))  # Flatten for inverse transform

# Reshape back to the original shape for plotting
y_test_inv = y_test_inv.reshape(y_test.shape)
predictions_inv = predictions_inv.reshape(predictions.shape)

# Plot forecasting
plot_forecasting(y_test_inv, predictions_inv)

