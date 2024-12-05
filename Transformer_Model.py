# Transformer_Model.py

import numpy as np
import tensorflow as tf
from keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout, Input
from keras.models import Model
from keras.optimizers import Adam
from keras.regularizers import l2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Positional Encoding function for time-series data
def positional_encoding(sequence_length, d_model):
    """
    Compute positional encoding for the sequence data.
    
    Args:
        sequence_length (int): Length of the input sequence.
        d_model (int): Dimensionality of the model.
    
    Returns:
        tf.Tensor: Positional encoding matrix as a TensorFlow tensor.
    """
    angles = np.arange(sequence_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
    
    pos_encoding = np.zeros((sequence_length, d_model))
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])  # Sin for even indices
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])  # Cos for odd indices
    
    # Convert the positional encoding to a TensorFlow tensor
    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)

# Transformer Block definition
# Transformer Block definition
def transformer_block(input_layer, attention_layer, normalization_layer1, normalization_layer2, 
                      dense_layer1, dropout_layer1, dense_layer2, dropout_layer2, dropout_rate):
    # Attention mechanism with dropout and residual connection
    attention_output = attention_layer(input_layer, input_layer)
    attention_output = dropout_layer1(attention_output)  # Apply first dropout
    attention_output = normalization_layer1(input_layer + attention_output)

    # Feed-forward network with dropout and residual connection
    ff_output = dense_layer1(attention_output)
    ff_output = dropout_layer1(ff_output)  # Apply dropout after first dense layer
    ff_output = dense_layer2(ff_output)
    ff_output = dropout_layer2(ff_output)  # Apply dropout after second dense layer
    output = normalization_layer2(attention_output + ff_output)
    
    return output

# Transformer Model Definition
class TransformerModel(Model):
    def __init__(self, num_heads, num_layers, seq_length, num_features, dropout_rate=0.4):
        super(TransformerModel, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.d_model = num_features

        # Input layer
        self.input_layer = Input(shape=(seq_length, num_features))

        # Positional encoding
        self.pos_encoding = positional_encoding(seq_length, self.d_model)

        # MultiHeadAttention layer
        self.attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=self.d_model)

        # LayerNormalization layers
        self.normalization_layer1 = LayerNormalization(epsilon=1e-6)
        self.normalization_layer2 = LayerNormalization(epsilon=1e-6)

        # Dense layers
        self.dense_layer1 = Dense(256, activation='relu')  # No L2 regularizer
        self.dense_layer2 = Dense(self.d_model)

        # Dropout layers
        self.dropout_layer1 = Dropout(dropout_rate)  # Dropout after dense_layer1
        self.dropout_layer2 = Dropout(dropout_rate)  # Dropout after dense_layer2

        # Transformer blocks
        self.transformer_blocks = [transformer_block for _ in range(num_layers)]

        # Output layer
        self.dense = Dense(1)

    def call(self, inputs):
        # Add positional encoding
        x = inputs + self.pos_encoding

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(
                x,
                self.attention_layer,
                self.normalization_layer1,
                self.normalization_layer2,
                self.dense_layer1,
                self.dropout_layer1,  # Use the correct dropout attribute
                self.dense_layer2,
                self.dropout_layer2,  # Use the correct dropout attribute
                0.1,
            )

        # Predict based on the last time step
        return self.dense(x[:, -1, :])

    def get_config(self):
        return {
            "num_heads": 2,
            "d_model": self.d_model,
            "ff_units": 64,
            "num_layers": len(self.transformer_blocks),
            "seq_length": self.seq_length,
            "dropout_rate": 0.4,
        }

# Build the Transformer model
def build_transformer_model(input_shape):
    """
    Build and compile the Transformer model.
    
    Args:
        input_shape (tuple): Shape of the input data (sequence_length, num_features).
    
    Returns:
        Model: Compiled Transformer model.
    """
    seq_length, num_features = input_shape  # Extract num_features from input_shape
    
    num_heads = 2  # Number of attention heads
    num_layers = 2 # Number of transformer blocks

        # Adjust the number of outputs here to match the number of neurons
    num_neurons = num_features   # Assuming num_features corresponds to the number of neurons
    
    model = TransformerModel(num_heads, num_layers, seq_length, num_features)

        # Change the output Dense layer to match the number of neurons (num_neurons)
    model.dense = Dense(num_neurons)  # Adjust the output size to match the number of neurons

    optimizer = Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])
    
    return model

