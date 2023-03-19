'''
MIT License
Copyright (c) 2022 Thomas Berthiaume
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from data.manage_data import load_autoenc_vectorizer

#from data.manage_data import load_autoenc_vectorizer, preprocess_text

starting_prompt = """
 |  __ \     | | |        \ \    / /_ |
 | |  | | ___| | |_ __ _   \ \  / / | |
 | |  | |/ _ \ | __/ _` |   \ \/ /  | |
 | |__| |  __/ | || (_| |    \  /   | |
 |_____/ \___|_|\__\__,_|     \/    |_|

    © 2021 - 2023 Thomas Berthiaume - All Rights Reserved.
"""


MAXLEN = 30
MAXVOCAB = 10000
VOCAB = 7210

class EncoderTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads):
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
        self.supports_masking = True


    def call(self, inputs, mask=None): #change =None
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attn_output = self.att(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
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

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

class DecoderConditionnalBlock(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads):
        super(DecoderConditionnalBlock, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        #self attention
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        #trought_vector // condition attention
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
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, condition_emb, mask=None): # =None
        ### mask
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        ### attention layer 1: SELF
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        ### attetion layer 2: OTHERS
        conditionnal_tv = layers.concatenate(inputs=[encoder_outputs, condition_emb])
        attention_output_2 = self.attention_2(
            query=out_1,
            value=conditionnal_tv,
            key=conditionnal_tv,
            attention_mask = padding_mask
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

class DecoderTransformerBlock(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads):
        super(DecoderTransformerBlock, self).__init__()
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
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, mask=None): # =None
        ### mask
        causal_mask = self.get_causal_attention_mask(inputs)
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, :], dtype="int32")
            padding_mask = tf.minimum(padding_mask, causal_mask)

        ### attention layer 1
        attention_output_1 = self.attention_1(
            query=inputs, value=inputs, key=inputs, attention_mask=causal_mask
        )
        out_1 = self.layernorm_1(inputs + attention_output_1)

        ### attetion layer 2
        attention_output_2 = self.attention_2(
            query=out_1,
            value=encoder_outputs,
            key=encoder_outputs,
            attention_mask = padding_mask
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0,
        )
        return tf.tile(mask, mult)

embed_dim = 64
num_heads = 4
latent_dim = 256 

def create_model():
    encoder_inputs = keras.Input(shape=(MAXLEN,), dtype="int32", name="encoder_inputs")
    x = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(encoder_inputs)
    encoder_outputs = EncoderTransformerBlock(embed_dim, latent_dim, num_heads)(x)
    #encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    condition_inputs = keras.Input(shape=(None, embed_dim), name="condition_inputs")
    x = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(decoder_inputs)
    #condition_emb = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(condition_inputs)
    #x = DecoderTransformerBlock(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
    x = DecoderConditionnalBlock(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs, condition_inputs) #condition_emb
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(VOCAB, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs, condition_inputs], decoder_outputs, name="output")

    decoder_outputs = decoder([decoder_inputs, encoder_outputs, condition_inputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs, condition_inputs], decoder_outputs, name="model3"
    )
    return transformer

def save_model(model: keras.Model, path):
    model.save_weights(str(path))

def load_model(path):
    load_auto_enc = create_model()
    load_auto_enc.load_weights(path)
    return load_auto_enc


def train():

    auto_enc = create_model()
    print(auto_enc.summary())
    #auto_enc.compile("adam", loss="categorical_crossentropy", metrics=["accuracy"])
    train_dataset, vectorizer = load_autoenc_vectorizer() # need the condition (emotion or context) input 
    print(train_dataset)
    #auto_enc.fit(train_dataset, epochs=20)
    #save_model(auto_enc, "./supreme_Delta")
    return vectorizer


if __name__ == '__main__':
    print(starting_prompt)
    vectorizer = train()
    ### DO NOT RUN THIS ON THIS PC! IT WILL EXPLODE
'''
### testing
VOCAB = vectorizer.get_vocabulary()

def decode_sentence(input_sentence):
    # Mapping the input sentence to tokens and adding start and end tokens
    input_sentence = tf.constant("[start] " + preprocess_text(input_sentence) + " [end]")
    up_dim = tf.expand_dims(input_sentence, -1)
    tokenized_input_sentence = vectorizer(
        up_dim
    )
    tokenized_input_sentence = tokenized_input_sentence[0]
    # Initializing the initial sentence consisting of only the start token.
    tokenized_target_sentence = tf.expand_dims(VOCAB.index("[start]"), 0)
    decoded_sentence = ""

    for i in range(MAXLEN):
        # Get the predictions
        predictions = auto_enc.predict(
            {
                "encoder_inputs": tf.expand_dims(tokenized_input_sentence, 0),
                "decoder_inputs": tf.expand_dims(
                    tf.pad(
                        tokenized_target_sentence,
                        [[0, MAXLEN - tf.shape(tokenized_target_sentence)[0]]],
                    ),
                    0,
                ),
            }
        )
        # Calculating the token with maximum probability and getting the corresponding word
        sampled_token_index = tf.argmax(predictions[0, i, :])
        sampled_token = VOCAB[sampled_token_index.numpy()]
        # If sampled token is the end token then stop generating and return the sentence
        if tf.equal(sampled_token_index, VOCAB.index("[end]")):
            break
        decoded_sentence += sampled_token + " "
        tokenized_target_sentence = tf.concat(
            [tokenized_target_sentence, [sampled_token_index]], 0
        )

    return decoded_sentence


sentence = decode_sentence("Hello")
print(sentence)
'''
