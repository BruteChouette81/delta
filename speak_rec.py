import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.python.keras.saving.save import load_model

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


class TransformerBlock1(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock1, self).__init__()
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

    def call(self, inputs, training): # training = True 
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output) # out2


class TokenAndPositionEmbedding1(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding1, self).__init__()
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

def new_model_sequential_speech():
    model = keras.Sequential([
        layers.Input(shape=(maxlen, ), name="speach_rec_input"), # name="emo_rec_input"
        TokenAndPositionEmbedding1(maxlen, vocab_size, embed_dim),
        TransformerBlock1(embed_dim, num_heads, ff_dim),
        layers.GlobalAveragePooling1D(name="pooling1D_speech"),
        layers.Dropout(0.1, name="dropout0_speech"),
        layers.Dense(20, activation="relu", name="dense0_speech"),
        layers.Dropout(0.1, name="dropout1_speech"),
        layers.Dense(3, activation="softmax", name="dense1_speech")
    ])
    return model


def new_model_test1(len_tags):
    inputs = layers.Input(name="speach_rec_input",shape=(maxlen, ))
    embedding_layer = TokenAndPositionEmbedding1(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock1(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D(name="pooling1D_speech")(x)
    x = layers.Dropout(0.1, name="dropout0_speech")(x)
    x = layers.Dense(20, activation="relu", name="dense0_speech")(x)
    x = layers.Dropout(0.1, name="dropout1_speech")(x)
    outputs = layers.Dense(3, activation="softmax", name="dense1_speech")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    #model.summary()
    #model.load_weights("weights/speak_rec")
    return model



def train():
    x_train, y_train, t, classes = prepare_train("ir.json")
    #x_train = x_train.reshape(-1, 18, 5)
    #y_train = y_train.reshape(-1, 18, 3)
    #model = new_model_test1(len(y_train[0])) old model 
    model = new_model_sequential_speech()
    model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=1, epochs=20)
    model.save("models_test/speak_rec")

def get_speech_model(top=True):
    model = load_model("models_test/speak_rec") # custom_objects = {"TokenAndPositionEmbedding": TokenAndPositionEmbedding, "TransformerBlock": TransformerBlock}
    if not top:
        model.pop() # check if model is sequential
        model.pop()
        return model

    else:
        return model

if __name__ == "__main__":
    #train()
    model = get_speech_model(True)
    model.summary()
    x_train, y_train, t, classes = prepare_train("ir.json")
    #x_train = x_train.reshape(-1, 18, 5)
    #y_train = y_train.reshape(-1, 18, 3)
    #model = new_model_test1(len(y_train[0])) old model 
    #model = new_model_sequential_speech()
    #model.summary()
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=1, epochs=2)
    
    #print(tf.__version__)