import tensorflow as tf
import struct
import numpy as np
import os
import time

path_to_file = tf.keras.utils.get_file(
    'shakespeare.txt',
    'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt'
)

# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print(f'Length of text: {len(text)} characters')

# The unique characters in the file
vocab = sorted(set(text))
ids_from_chars = tf.keras.layers.StringLookup(vocabulary=list(vocab), mask_token=None)

# training data
all_ids = ids_from_chars(tf.strings.unicode_split(text, 'UTF-8'))
ids_dataset = tf.data.Dataset.from_tensor_slices(all_ids)
seq_length = 100
sequences = ids_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(sequence):
    input_text = sequence[:-1]
    target_text = sequence[1:]
    return input_text, target_text

# Batch size
BATCH_SIZE = 64
BUFFER_SIZE = 10000
dataset = (
    sequences
    .map(lambda sequence: (sequence[:-1], sequence[1:]))
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE)
)

# Length of the vocabulary in StringLookup Layer
vocab_size = len(ids_from_chars.get_vocabulary())

# The embedding dimension
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

class MyModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            rnn_units, return_sequences=True, return_state=True
        )
        self.dense = tf.keras.layers.Dense(vocab_size)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state:
            return x, states
        else:
            return x

model = MyModel(
    vocab_size=vocab_size,
    embedding_dim=embedding_dim,
    rnn_units=rnn_units
)
loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

# Directory where the checkpoints will be saved
checkpoint_dir = '/content/drive/MyDrive/Colab Notebooks/research/rnn_text_gen/v1.ckpt'
# Name of the checkpoint files
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_dir,
    save_weights_only=True,
    save_best_only=True,
    monitor='loss'
)

# train the model
EPOCHS = 20
model.compile(optimizer='adam', loss=loss)
model.fit(
    dataset, epochs=EPOCHS, callbacks=[checkpoint_callback]
)
model.fit(
    dataset, epochs=10, callbacks=[checkpoint_callback]
)

# export the model to a binary file
f_output = open("/content/drive/MyDrive/Colab Notebooks/research/rnn_text_gen/gru.bin", "wb")

for layer in model.weights:
    layer_params = layer.numpy()
    print(f"Layer: {layer.name} with shape {layer_params.shape}")

    n_dims = len(layer_params.shape)
    f_output.write(struct.pack("i", n_dims))

    for i in range(n_dims):
        # because the order of dimensions are reversed in GGML
        f_output.write(struct.pack("i", layer_params.shape[n_dims - 1 - i]))
    layer_params.astype(np.float32).tofile(f_output)
f_output.close()

