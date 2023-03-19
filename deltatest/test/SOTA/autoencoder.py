
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras.models import Model

import numpy as np

from data.manage_data import load_autoenc_vectorizer


INMAXLEN = 30
OUTMAXLEN = 30
embed_dim = 16 # Embedding size for each token
num_heads = 4 # Number of attention heads
ff_dim = 128  # Hidden layer size in feed forward network inside transformer
batch_size = 32

class EncoderTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(EncoderTransformerBlock, self).__init__()
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


class TokenAndPositionEmbeddingModel3(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbeddingModel3, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

class DecoderTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads, **kwargs):
        super(DecoderTransformerBlock, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        self.dense_proj = keras.Sequential(
            [
                layers.Dense(latent_dim, activation="relu"), 
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()
        self.layernorm_3 = layers.LayerNormalization()

    def call(self, inputs, encoder_outputs):
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

#the model 
def build_model(maxlen, inp_vocab, out_vocab):

    text_inp = layers.Input(shape=(maxlen, ), name="text_inp") # name="emo_rec_input"
    x = TokenAndPositionEmbeddingModel3(maxlen, inp_vocab, embed_dim)(text_inp)
    encoder = EncoderTransformerBlock(embed_dim, num_heads, ff_dim)(x)
    encoder_model = Model(text_inp, encoder)

    decoder_inputs = layers.Input(shape=(None, ), name="decoder_inp")
    encoded_seq_inputs = layers.Input(shape=(None, embed_dim), name="encoded_seq_inputs")

    x = TokenAndPositionEmbeddingModel3(maxlen, out_vocab, embed_dim)(decoder_inputs)
    x = DecoderTransformerBlock(embed_dim, ff_dim, num_heads)(x, encoded_seq_inputs)
    #x = layers.GlobalAveragePooling1D()(x)
    dropout = layers.Dropout(0.5)(x)
    output = layers.Dense(out_vocab, activation="softmax")(dropout)

    decoder = Model([decoder_inputs, encoded_seq_inputs], output)
    output = decoder([decoder_inputs, encoder])

    model = Model(inputs=[text_inp, decoder_inputs], outputs=output, name="Model3")
    print(model.summary())
    return model

def train():
    #x_pdoc, x_vocab_size, x_tokenizer, y_pdoc, y_vocab_size, y_tokenizer, y_target_doc = load_autoenc_tokenizer()
    train_ds, vectorize = load_autoenc_vectorizer()

    model = build_model(INMAXLEN, len(vectorize.get_vocabulary()), len(vectorize.get_vocabulary())) # only if in-max lenght is = to out-max lenght

    model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    model.fit(train_ds, batch_size=batch_size, epochs=10)
    model.save("models/model3_10")

    return model, vectorize

badmodel, vectorize = train()

model = keras.models.load_model("models/model3_10")
sentence = "[start] hello how are you [end]"
tokenize_input = vectorize([sentence])
print(tokenize_input)
decoded_sentence = "[start] "
#decoded_list = ["start", "i"]

vocab_index = dict(zip(range(len(vectorize.get_vocabulary())), vectorize.get_vocabulary()))

for i in range(20):
        #sample_index = len(list(decoded_list)) - 1
        tokenized_target_sentence = vectorize([decoded_sentence])[:, :-1] #remove padding
        predictions = model([tokenize_input, tokenized_target_sentence])

        sampled_token_index = np.argmax(predictions[0, i, :])
        sampled_token = vocab_index[sampled_token_index] # no + 1

        decoded_sentence += " " + sampled_token
        #decoded_list.append(sampled_token)

        if sampled_token == "[end]":
            print("ending sequence")
            break

print(decoded_sentence)
