import keras
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

from test import encoded
from test import dataset

embed_dim = 4  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 16  # Hidden layer size in feed forward network inside transformer

maxlen = 5
vocab_size = 45


def prepare_train(data):
    index = dataset(f"data/{data}")
    bag, documents, t, classes = index.open_json()

    encoded_data = encoded(bag, documents, t, classes)
    train = encoded_data.encoded_doc()

    x_train = train[1]
    x_train = np.array(x_train)
    y_train = train[0]
    y_train = np.array(y_train)

    return x_train, y_train, t, classes


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # out2


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

    


def new_model_test(len_tags):
    inputs = layers.Input(shape=(maxlen, ))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(3, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



class model_emotion:
    def __init__(self, data):
        self.data = data

    def model2(self):
        x_train, y_train, t, classes = prepare_train(self.data)
        model = keras.Sequential([
            keras.layers.Embedding(45, 128),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(len(y_train[0]), activation='sigmoid')
        ])
        return model, x_train, y_train, t



def train():
    x_train, y_train, t, classes = prepare_train("emotion_pattern.json")
    print(len(y_train[0]))
    model = new_model_test(len(y_train[0]))
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train, y_train, batch_size=1, epochs=20
    )
    '''
    model2, x_emotion_train, y_emotion_train, t2 = model_emotion("emotion_pattern.json").model2() # make dataset a parameter
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #make loss and opt. params
    model2.fit(x_emotion_train, y_emotion_train, epochs=100, batch_size=2, verbose=1) # amke epoch and batch params
    model2.save("../server_test/models/model2.h5")
    '''

train()