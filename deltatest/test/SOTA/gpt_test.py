import time
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.models import Model

import numpy as np

from data.manage_data import load_generative



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
        self.dropout1 = layers.Dropout(rate, name="dropout_encoder")
        self.dropout2 = layers.Dropout(rate, name="dropout_encoder1")

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

 # Only consider the top 20k words
maxlen = 80  # Max sequence size
embed_dim = 64  # Embedding size for each token
num_heads = 4  # Number of attention heads
feed_forward_dim = 256  # Hidden layer size in feed forward network inside transformer
batch_size = 128

callbacks = []


def create_model(vocab_size):
    inputs = layers.Input(shape=(maxlen,), dtype=tf.int32)
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, feed_forward_dim)
    x = transformer_block(x)
    outputs = layers.Dense(vocab_size)(x)
    model = keras.Model(inputs=inputs, outputs=[outputs, x])
    return model

class TextGenerator(keras.callbacks.Callback):
    """A callback to generate text from a trained model.

    Arguments:
        max_tokens: Integer, the number of tokens to be generated after prompt.
        start_tokens: List of integers, the token indices for the starting prompt.
        index_to_word: List of strings, obtained from the TextVectorization layer.
        top_k: Integer, sample from the `top_k` token predictions.
        print_every: Integer, print after this many epochs.
    """

    def __init__(
        self, max_tokens, start_tokens, index_to_word, top_k=10, print_every=1
    ):
        self.max_tokens = max_tokens
        self.start_tokens = start_tokens
        self.index_to_word = index_to_word
        self.print_every = print_every
        self.k = top_k

    def sample_from(self, logits):
        logits, indices = tf.math.top_k(logits, k=self.k, sorted=True)
        indices = np.asarray(indices).astype("int32")
        preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
        preds = np.asarray(preds).astype("float32")
        return np.random.choice(indices, p=preds)

    def detokenize(self, number):
        return self.index_to_word[number]

    def on_epoch_end(self, epoch, logs=None):
        start_tokens = [_ for _ in self.start_tokens]
        if (epoch + 1) % self.print_every != 0:
            return
        num_tokens_generated = 0
        tokens_generated = []
        while num_tokens_generated <= self.max_tokens:
            pad_len = maxlen - len(start_tokens)
            sample_index = len(start_tokens) - 1
            if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
            elif pad_len > 0:
                x = start_tokens + [0] * pad_len
            else:
                x = start_tokens
            x = np.array([x])
            y, _ = self.model.predict(x)
            sample_token = self.sample_from(y[0][sample_index])
            tokens_generated.append(sample_token)
            start_tokens.append(sample_token)
            num_tokens_generated = len(tokens_generated)
        txt = " ".join(
            [self.detokenize(_) for _ in self.start_tokens + tokens_generated]
        )
        print(f"generated text:\n{txt}\n")
        callbacks.append(txt)

def train(epochs):
    filename = ["data/generative_data3.txt", "data/empathetic_dialogues.txt"]
    text_ds = tf.data.TextLineDataset(filename)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)
    vocab, text_ds = load_generative(text_ds)

    word_to_index = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index # dict of {"WordAsStr": IndexOfWord}

    max_token = 80
    #prompt = "start hey"
    special_tokens = ["<start", "<end"]
    start_tokens = [8, 4, 365, 16, 65, 23, 82, 66, 19] #[word_to_index.get(_, 1) for _ in prompt.split()]
    print(start_tokens)

    special_tokens_ids = [word_to_index.get(_, 1) for _ in special_tokens]
    print(special_tokens_ids)

    text_gen_callback = TextGenerator(max_token, start_tokens, vocab)
    model = create_model(len(vocab))

    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    model.compile(optimizer="adam", loss=[loss_fn, None])

    model.fit(text_ds, verbose=1, epochs=epochs, callbacks=[text_gen_callback])
    return model, callbacks

def sample_from(logits, k):
    logits, indices = tf.math.top_k(logits, k=k, sorted=True)
    indices = np.asarray(indices).astype("int32")
    preds = keras.activations.softmax(tf.expand_dims(logits, 0))[0]
    preds = np.asarray(preds).astype("float32")
    return np.random.choice(indices, p=preds)

def detokenize(index_to_word, number):
        return index_to_word[number]

def test(model, maxlen, input, index_to_word):
    start_tokens = [_ for _ in input]
    num_tokens_generated = 0
    tokens_generated = []

    while num_tokens_generated <= maxlen:
        pad_len = maxlen - len(start_tokens)
        sample_index = len(start_tokens) - 1

        if pad_len < 0:
                x = start_tokens[:maxlen]
                sample_index = maxlen - 1
        elif pad_len > 0:
            x = start_tokens + [0] * pad_len
        else:
            x = start_tokens
        x = np.array([x])
        y, _ = model.predict(x)
        sample_token = sample_from(y[0][sample_index], 10)
        tokens_generated.append(sample_token)
        start_tokens.append(sample_token)
        num_tokens_generated = len(tokens_generated)
    txt = " ".join(
        [detokenize(index_to_word, _) for _ in start_tokens + tokens_generated]
    )
    return txt




if __name__ == '__main__':
    
    model, callbacks = train(5)
    #model.save('model4_10')

    #time.sleep(5)
    #model, callbacks = train(25)
    #model.save('model4_25')

    '''
    filename = ["generative_data3.txt"]
    text_ds = tf.data.TextLineDataset(filename)
    text_ds = text_ds.shuffle(buffer_size=256)
    text_ds = text_ds.batch(batch_size)
    vocab, text_ds = load_generative(text_ds)

    word_to_index = {}
    for index, word in enumerate(vocab):
        word_to_index[word] = index # dict of {"str": index}

    #prompt = "start hi how are you doing fine and you i am good"
    #start_tokens = [word_to_index.get(_, 1) for _ in prompt.split()]
    #print(start_tokens)
    start_tokens = [25, 12, 48, 19, 22, 14, 30, 78, 5]
    
    model = keras.models.load_model("model4_10")
    response = test(model, maxlen, start_tokens, vocab)
    print(response)
    '''
