# Transformer_Model.py

import numpy as np
import tensorflow as tf
from keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout, Input, Flatten
from keras.models import Model
from keras import layers, models
from keras.optimizers import Adam
from keras.regularizers import l2
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Positional Encoding function for time-series data
def positional_encoding(sequence_length, d_model):
    angles = np.arange(sequence_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
    pos_encoding = np.zeros((sequence_length, d_model))
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)

# Transformer Block definition
def transformer_block(input_layer, attention_layer, normalization_layer1, normalization_layer2, 
                      dense_layer1, dropout_layer1, dense_layer2, dropout_layer2):
    attention_output = attention_layer(input_layer, input_layer)
    attention_output = dropout_layer1(attention_output)
    attention_output = normalization_layer1(input_layer + attention_output)

    ff_output = dense_layer1(attention_output)
    ff_output = dropout_layer1(ff_output)
    ff_output = dense_layer2(ff_output)
    ff_output = dropout_layer2(ff_output)
    output = normalization_layer2(attention_output + ff_output)
    
    return output

# Transformer Model Definition
class TransformerModel(Model):
    def __init__(self, num_heads, num_layers, seq_length, num_features, dropout_rate=0.4):
        super(TransformerModel, self).__init__()
        self.seq_length = seq_length
        self.num_features = num_features
        self.d_model = num_features

        self.input_layer = Input(shape=(seq_length, num_features))
        self.pos_encoding = positional_encoding(seq_length, self.d_model)
        self.attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=self.d_model)
        self.normalization_layer1 = LayerNormalization(epsilon=1e-6)
        self.normalization_layer2 = LayerNormalization(epsilon=1e-6)
        self.dense_layer1 = Dense(64, activation='relu')
        self.dense_layer2 = Dense(self.d_model)
        self.dropout_layer1 = Dropout(dropout_rate)
        self.dropout_layer2 = Dropout(dropout_rate)
        self.transformer_blocks = [transformer_block for _ in range(num_layers)]
        self.dense = Dense(future_steps)

    def call(self, inputs):
        x = inputs + self.pos_encoding
        for block in self.transformer_blocks:
            x = block(x, self.attention_layer, self.normalization_layer1, self.normalization_layer2,
                      self.dense_layer1, self.dropout_layer1, self.dense_layer2, self.dropout_layer2)
        return self.dense(x[:, -1, :])

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
def build_transformer_model(input_shape, future_steps):
    # Input layer
    input_layer = layers.Input(shape=input_shape)
    
    # Layer normalization
    x = layers.LayerNormalization()(input_layer)
    
    # Multi-head attention layer
    x = layers.MultiHeadAttention(num_heads=2, key_dim=64)(x, x)
    x = layers.Dropout(0.1)(x)
    
    # Second layer normalization
    x = layers.LayerNormalization()(x)
    
    # Dense layer
    x = layers.Dense(64)(x)
    x = layers.Dropout(0.1)(x)
    
    # Flatten layer to match output shape
    x = layers.Flatten()(x)
    
    # Output layer to match (future_steps, n_features)
    output_layer = layers.Dense(future_steps * input_shape[-1])(x)  # Adjust for number of features
    
    # Reshape to (future_steps, n_features)
    output_layer = layers.Reshape((future_steps, input_shape[-1]))(output_layer)

    # Build the model
    model = models.Model(inputs=input_layer, outputs=output_layer)
    
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])  # Add 'mae' here
    
    return model

