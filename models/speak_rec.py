import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

from test import dataset
from test import encoded

embed_dim = 4  # Embedding size for each token
num_heads = 2  # Number of attention heads
ff_dim = 16  # Hidden layer size in feed forward network inside transformer

maxlen = 5
vocab_size = 33


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

    def call(self, inputs, training = True):
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

    
class model_speech_rec:
    def __init__(self, data):
        self.data = data

    def model1(self):
        x_train, y_train, t, classes = prepare_train(self.data)
        model = keras.Sequential([
            keras.layers.Embedding(33, 128),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(len(y_train[0]), activation='sigmoid')
        ])
        return model, x_train, y_train, t


def new_model_test1(len_tags):
    inputs = layers.Input(name="speach_rec_input",shape=(maxlen, ))
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
    #model.summary()
    model.load_weights("model_test1/speak_rec")
    return model



def train():
    x_train, y_train, t, classes = prepare_train("ir.json")
    #x_train = x_train.reshape(-1, 18, 5)
    #y_train = y_train.reshape(-1, 18, 3)
    model = new_model_test1(len(y_train[0]))
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(x_train, y_train, batch_size=1, epochs=1)
    #model.save_weights("model_test1/speak_rec")

if __name__ == "__main__":
    train()