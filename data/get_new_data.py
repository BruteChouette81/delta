from datasets import load_dataset
from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers


dataset = load_dataset("poem_sentiment")
#print(dataset)
embed_dim = 4  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer

maxlen = 25

def process_x_dataset(data):
    bag = []
    for sequence in dataset['train']['verse_text']:
        bag.append(sequence)

    for sequence in dataset['test']['verse_text']:
        bag.append(sequence)

    for sequence in dataset['validation']['verse_text']:
        bag.append(sequence)

    t = Tokenizer()
    t.fit_on_texts(bag)

    doc = t.texts_to_sequences(bag)
    p_doc = keras.preprocessing.sequence.pad_sequences(doc, maxlen=25, padding="post")
    vocab_size = len(t.word_index) + 1
    return p_doc, vocab_size, t


def process_y_dataset(data):
    bag = []
    classes = ["0", "1", "2", "3"]
    for label in dataset['train']['label']:
        bag.append(str(label))

    for label in dataset['test']['label']:
        bag.append(str(label))

    for label in dataset['validation']['label']:
        bag.append(str(label))


    out_empty = [0] * 4
    training = []

    for labels in bag:
        output = list(out_empty)
        output[classes.index(labels)] = 1
        training.append(output)
    return training

xdoc, vocab_size, t = process_x_dataset(dataset)
x_train = np.array(xdoc)
ydoc = process_y_dataset(dataset)
y_train = np.array(ydoc)


''''test new model... not definitive'''

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

    def call(self, inputs, training): #change =True 
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
    inputs = layers.Input(name="emo_rec_input", shape=(maxlen, ))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(4, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = new_model_test(4)
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=16, epochs=15)

test = ["this is sad"]
numericls = t.texts_to_sequences(test)
print(numericls)
#p_test_doc = np.array(numericls).reshape(1, len(numericls))
x_test_predict = keras.preprocessing.sequence.pad_sequences(numericls, maxlen=25, padding="post")
print(x_test_predict)

y_test_predict = model.predict(x_test_predict)
print(y_test_predict)



