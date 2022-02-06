from datasets import load_dataset
from keras.preprocessing.text import Tokenizer
import numpy as np
from tensorflow import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model


### add from datasets import load_dataset
dataset = load_dataset("emotion")
### and combine those data set

#dataset = load_dataset("poem_sentiment")
#print(dataset)
embed_dim = 4  # Embedding size for each token
num_heads = 8  # Number of attention heads
ff_dim = 64  # Hidden layer size in feed forward network inside transformer

maxlen = 25

def process_x_dataset_poem(dataset):
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


def process_y_dataset_poem(dataset):
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

def process_x_dataset_emotion(dataset):
    bag = []
    num_sentence = 0
    for sequence in dataset['train']['text']:
        bag.append(str(sequence))

        if num_sentence == 2000:
            print("stopping the loop")
            break

        else:
            num_sentence += 1

    t = Tokenizer()
    t.fit_on_texts(bag)

    doc = t.texts_to_sequences(bag)
    p_doc = keras.preprocessing.sequence.pad_sequences(doc, maxlen=25, padding="post")
    vocab_size = len(t.word_index) + 1
    return p_doc, vocab_size, t

def process_y_dataset_emotion(dataset):
    bag = []
    classes = ["0", "1", "2", "3", "4", "5"]
    num_labels = 0 
    for label in dataset['train']['label']:
        bag.append(str(label))

        if num_labels == 2000:
            print("stopping the loop")
            break

        else:
            num_labels += 1

    out_empty = [0] * 6
    training = []

    for labels in bag:
        output = list(out_empty)
        output[classes.index(labels)] = 1
        training.append(output)
    return training


### poem 
#xdoc, vocab_size, t = process_x_dataset_poem(dataset)
#x_train = np.array(xdoc)
#ydoc = process_y_dataset_poem(dataset)
#y_train = np.array(ydoc)

xdoc, vocab_size, t = process_x_dataset_emotion(dataset)
x_train = np.array(xdoc)

ydoc = process_y_dataset_emotion(dataset)
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


def new_model_sequential():
    model = keras.Sequential([
        layers.Input(shape=(maxlen, ), name="emo_rec_input"), # name="emo_rec_input"
        TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim),
        TransformerBlock(embed_dim, num_heads, ff_dim),
        layers.GlobalAveragePooling1D(),
        layers.Dropout(0.1),
        layers.Dense(40, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(6, activation="softmax")
    ])
    return model
'''
model = new_model_sequential()
model.summary()
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, batch_size=32, epochs=20)
model.save("emotion_new_data")
'''
model = load_model("models_test/emotion_new_data")
test = ["i have seen heard and read over the past couple of days i am left feeling impressed by more than a few companies"] 
numericls = t.texts_to_sequences(test)
print(numericls)
#p_test_doc = np.array(numericls).reshape(1, len(numericls))
x_test_predict = keras.preprocessing.sequence.pad_sequences(numericls, maxlen=25, padding="post")
print(x_test_predict)

y_test_predict = model.predict(x_test_predict)
print(np.argmax(y_test_predict)) #sadness = 0, joy = 1, love = 2, anger = 3, fear = 4, surprise = 5

