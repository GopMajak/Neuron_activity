import scipy.io as sio
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_mat_file(file_path):
    try:
        mat_data = sio.loadmat(file_path)
        logger.info("Successfully loaded .mat file.")
        return mat_data
    except Exception as e:
        logger.error(f"Error loading .mat file: {e}")
        raise

def sequence_target(N_activity, seq_length, future_steps):

    X, y = [], []
    for i in range(len(N_activity) - seq_length - future_steps):  # Adjust for future_steps
        X.append(N_activity[i:i+seq_length])
        y.append(N_activity[i+seq_length:i+seq_length+future_steps])  # Predict future_steps ahead
    
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    # Reshape y if it's a 1D array
    if y.ndim == 1:
        y = y.reshape(-1, 1)
    
    return X, y


def preprocess_data(mat_data, seq_length, future_steps):

    try:
        activity_data = mat_data['spk_arr']
        df = pd.DataFrame(activity_data, columns=[f'neuron{i + 1}' for i in range(activity_data.shape[1])])
        N_activity = df.values  # Use .values to get numpy array
        logger.info("DataFrame created and converted to numpy array.")
    except KeyError as e:
        logger.error(f"KeyError accessing 'spk_arr': {e}")
        raise
    except Exception as e:
        logger.error(f"Error in data processing: {e}")
        raise

    # Use future_steps in the sequence_target function to create sequences
    X, y = sequence_target(N_activity, seq_length, future_steps)
    logger.info(f"Sequences created. Shapes - X: {X.shape}, y: {y.shape}")

    if X.size > 0 and y.size > 0:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        logger.info("Data split into training and testing sets.")
        return X_train, X_test, y_train, y_test
    else:
        logger.error("Data is not suitable for splitting. Check the data preparation steps.")
        raise ValueError("Data is not suitable for splitting. Check the data preparation steps.")
