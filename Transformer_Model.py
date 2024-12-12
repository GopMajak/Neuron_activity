import numpy as np
import tensorflow as tf
from keras.layers import MultiHeadAttention, Dense, LayerNormalization, Dropout, Input, Flatten, Reshape, Embedding
from keras.models import Model
from keras import layers
from keras.optimizers import Adam

# Positional Encoding function
def positional_encoding(sequence_length, d_model):
    angles = np.arange(sequence_length)[:, np.newaxis] / np.power(10000, (2 * (np.arange(d_model)[np.newaxis, :] // 2)) / d_model)
    pos_encoding = np.zeros((sequence_length, d_model))
    pos_encoding[:, 0::2] = np.sin(angles[:, 0::2])
    pos_encoding[:, 1::2] = np.cos(angles[:, 1::2])
    return tf.convert_to_tensor(pos_encoding, dtype=tf.float32)

# Transformer Block definition
def transformer_block(input_layer, num_heads, d_model, ff_units, dropout_rate):
    attention_layer = MultiHeadAttention(num_heads=num_heads, key_dim=d_model)
    normalization_layer1 = LayerNormalization(epsilon=1e-6)
    normalization_layer2 = LayerNormalization(epsilon=1e-6)
    dense_layer1 = Dense(ff_units, activation='relu')
    dense_layer2 = Dense(d_model)
    dropout_layer = Dropout(dropout_rate)

    # Self-attention
    attention_output = attention_layer(input_layer, input_layer)
    attention_output = dropout_layer(attention_output)
    attention_output = normalization_layer1(input_layer + attention_output)

    # Feed-forward network
    ff_output = dense_layer1(attention_output)
    ff_output = dropout_layer(ff_output)
    ff_output = dense_layer2(ff_output)
    ff_output = dropout_layer(ff_output)
    output = normalization_layer2(attention_output + ff_output)

    return output

# Transformer Model Definition
class TransformerModel(Model):
    def __init__(self, seq_length, d_model, num_heads, num_layers, ff_units, dropout_rate, future_steps, vocab_size):
        super(TransformerModel, self).__init__()
        self.seq_length = seq_length
        self.d_model = d_model
        self.future_steps = future_steps
        self.pos_encoding = positional_encoding(seq_length, d_model)

        # Embedding layer for input indices
        self.embedding = Embedding(input_dim=vocab_size, output_dim=d_model)

        # Transformer blocks
        self.transformer_blocks = [
            lambda x: transformer_block(x, num_heads, d_model, ff_units, dropout_rate)
            for _ in range(num_layers)
        ]
        self.output_dense = Dense(future_steps * d_model)

    def call(self, inputs):
        # Apply embedding
        x = self.embedding(inputs)

        # Add positional encoding
        x += self.pos_encoding

        # Pass through Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Flatten and project to output
        x = Flatten()(x)
        x = self.output_dense(x)

        # Reshape to match (future_steps, d_model)
        x = Reshape((self.future_steps, self.d_model))(x)

        return x

# Model instantiation
def build_transformer_model(input_shape, future_steps, vocab_size, num_heads=2, num_layers=2, ff_units=64, dropout_rate=0.1):
    seq_length, d_model = input_shape

    # Build model
    model = TransformerModel(
        seq_length=seq_length,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        ff_units=ff_units,
        dropout_rate=dropout_rate,
        future_steps=future_steps,
        vocab_size=vocab_size
    )

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error', metrics=['mae'])

    return model
