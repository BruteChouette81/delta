'''
MIT License
Copyright (c) 2023 Thomas Berthiaume
'''

import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from data.getData import load_crystal_vectorizer, preprocess_text

#from data.manage_data import load_autoenc_vectorizer

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


starting_prompt = """
   ____                _        _  __     ___ 
  / ___|_ __ _   _ ___| |_ __ _| | \ \   / / |
 | |   | '__| | | / __| __/ _` | |  \ \ / /| |
 | |___| |  | |_| \__ \ || (_| | |   \ V / | |
  \____|_|   \__, |___/\__\__,_|_|    \_/  |_|
             |___/                            

    © 2023 - 2024 Thomas Berthiaume - All Rights Reserved.
"""

MAXLEN = 30
MAXVOCAB = 40000
VOCAB = 16081

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

class DeltaEncoderBlock(layers.Layer):
    def __init__(self, embed_dim, ff_dim, num_heads):
        super(EncoderTransformerBlock, self).__init__()
        self.att1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.att2 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = layers.LayerNormalization(epsilon=1e-6)
        self.supports_masking = True


    def call(self, inputs, emotion_emb, mask=None): #change =None
        if mask is not None:
            padding_mask = tf.cast(mask[:, tf.newaxis, tf.newaxis, :], dtype="int32")
        attn_output = self.att1(
            query=inputs, value=inputs, key=inputs, attention_mask=padding_mask
        )
        out1 = self.layernorm1(inputs + attn_output)

        conditionnal_tv = layers.concatenate(inputs=[inputs, emotion_emb])
        attn_output2 = self.att2(
            query=out1,
            value=conditionnal_tv, 
            key=conditionnal_tv,
            attention_mask=padding_mask
        )

        out2 = self.layernorm2(out1 + attn_output2)
        
        
        ffn_output = self.ffn(out2)
        return self.layernorm2(out2 + ffn_output) # out3



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



class DeltaDecoderBlock(layers.Layer):
    def __init__(self, embed_dim, latent_dim, num_heads):
        super(DeltaDecoderBlock, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads

        #self attention
        self.attention_1 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        #trought_vector // emotion attention
        self.attention_2 = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim
        )
        #trought_vector // context attention
        self.attention_3 = layers.MultiHeadAttention(
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
        self.layernorm_4 = layers.LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, emotion_emb, context_emb, mask=None): # =None
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

        ### attention layer 2: EMOTION SIGNAL
        conditionnal_tv = layers.concatenate(inputs=[encoder_outputs, emotion_emb])
        attention_output_2 = self.attention_2(
            query=out_1,
            value=conditionnal_tv,
            key=conditionnal_tv,
            attention_mask = padding_mask
        )
        out_2 = self.layernorm_2(out_1 + attention_output_2)

        ### attention layer 2: CONTEXT
        conditionnal_tv2 = layers.concatenate(inputs=[encoder_outputs, context_emb]) #context with input and no outputs
        attention_output_3 = self.attention_3(
            query=out_2, #out_1
            value=conditionnal_tv2,
            key=conditionnal_tv2,
            attention_mask = padding_mask
        )
        out_3 = self.layernorm_3(out_2 + attention_output_3)
        

        proj_output = self.dense_proj(out_3)
        return self.layernorm_4(out_3 + proj_output)

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


def model1(): #decode base on condition
    encoder_inputs = keras.Input(shape=(MAXLEN,), dtype="int32", name="encoder_inputs")
    x = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(encoder_inputs)
    encoder_outputs = EncoderTransformerBlock(embed_dim, latent_dim, num_heads)(x)
    #encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(MAXLEN,), dtype="int32", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    emotion_inputs = keras.Input(shape=(MAXLEN,), name="emotion_inputs")
    condition_inputs = keras.Input(shape=(MAXLEN,), name="condition_inputs")
    
    x = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(decoder_inputs)
    #encoded_embed = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(encoded_seq_inputs)
    condition_emb = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(condition_inputs) # 1, 6
    
    emotion_emb = layers.Embedding(6, 64, input_length=30)(emotion_inputs)
    #emotion_emb = layers.Reshape((30, 64), input_shape=(1, 64))(emotion_emb)
    

    print("embedding shape emotions: ")
    print(emotion_emb.shape)
    #x = DecoderTransformerBlock(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
    x = DeltaDecoderBlock(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs, emotion_emb, condition_emb) #emotion_emb, condition_emb
    print(x.shape)
    drop = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(VOCAB, activation="softmax", name="output_dense")(drop)
    print(decoder_outputs.shape)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs, emotion_inputs, condition_inputs], decoder_outputs, name="output")

    decoder_outputs = decoder([decoder_inputs, encoder_outputs, emotion_inputs, condition_inputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs, emotion_inputs, condition_inputs], decoder_outputs, name="crystalModel1"
    )
    return transformer

def model2(): #encode emotions
    encoder_inputs = keras.Input(shape=(MAXLEN,), dtype="int32", name="encoder_inputs")
    emotion_inputs = keras.Input(shape=(None, embed_dim), dtype="int32", name="encoder_inputs")
    x = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(encoder_inputs)
    encoder_outputs = DeltaEncoderBlock(embed_dim, latent_dim, num_heads)(x, emotion_inputs) #need to embed them
    #encoder = keras.Model(encoder_inputs, encoder_outputs)

    decoder_inputs = keras.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    encoded_seq_inputs = keras.Input(shape=(None, embed_dim), name="decoder_state_inputs")
    emotion_inputs = keras.Input(shape=(None, embed_dim), name="emotion_inputs") #emotion can be remove of the condition since they are in the encodings
    condition_inputs = keras.Input(shape=(None, embed_dim), name="condition_inputs")
    
    x = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(decoder_inputs)
    #condition_emb = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(condition_inputs)
    #emotion_emb = TokenAndPositionEmbeddingModel3(MAXLEN, VOCAB, embed_dim)(emotion_inputs)
    #x = DecoderTransformerBlock(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs)
    x = DeltaDecoderBlock(embed_dim, latent_dim, num_heads)(x, encoded_seq_inputs, emotion_inputs, condition_inputs) #emotion_emb, condition_emb
    x = layers.Dropout(0.5)(x)
    decoder_outputs = layers.Dense(VOCAB, activation="softmax")(x)
    decoder = keras.Model([decoder_inputs, encoded_seq_inputs, emotion_inputs, condition_inputs], decoder_outputs, name="output")

    decoder_outputs = decoder([decoder_inputs, encoder_outputs, condition_inputs])
    transformer = keras.Model(
        [encoder_inputs, decoder_inputs, emotion_inputs, condition_inputs], decoder_outputs, name="Crystal - model1"
    )
    return transformer

def save_model(model: keras.Model, path):
    model.save_weights(str(path))

def load_model(path):
    load_auto_enc = model1()
    load_auto_enc.load_weights(path)
    return load_auto_enc

def train():
    model = model1()
    print(model.summary())
    model.compile("adam", loss='sparse_categorical_crossentropy', metrics=["accuracy"])
    dataset, vectorizer = load_crystal_vectorizer()
    model.fit(dataset, epochs=100)
    save_model(model, "./deltatest/weights/crystalModel1") # accuracy: 0.6413, loss: 0.8284, No gradient desapearing


def predict(input_sentence, context):
    model = load_model("./deltatest/weights/crystalModel1")
    
    dataset, vectorizer = load_crystal_vectorizer()
    VOCAB = vectorizer.get_vocabulary()

    #### ENCODING
    # Mapping the input sentence to tokens and adding start and end tokens
    input_sentence = tf.constant(preprocess_text(input_sentence)) # add "[start] + + [end]" or remove entire line
    up_dim = tf.expand_dims(input_sentence, -1)
    tokenized_input_sentence = vectorizer(
        up_dim
    )
    tokenized_input_sentence = tokenized_input_sentence[0]

    #emotion: 
    tokenized_input_emotions = np.array([1] *30) #get the real emotion code for the encription

    #context:

    if context: 
        start_context = context
    else: 
        start_context = [""]

    tokenized_input_context = vectorizer(
        start_context
    )

    #### DECODING
    # Initializing the initial sentence consisting of only the start token.
    tokenized_target_sentence = tf.expand_dims(VOCAB.index("[start]"), 0)
    decoded_sentence = ""

    for i in range(MAXLEN):
        # Get the predictions
        predictions = model.predict(
            {
                "encoder_inputs": tf.expand_dims(tokenized_input_sentence, 0), #remove expand dims
                "emotion_inputs": tokenized_input_emotions,
                "condition_inputs": tokenized_input_context,
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


#sentence = predict("Hello")
#print(sentence)

if __name__ == '__main__':
    print(starting_prompt)
    train()