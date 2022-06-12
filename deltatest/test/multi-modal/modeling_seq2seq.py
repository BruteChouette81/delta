import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model



def encoder(vocab_size, embedding_size):
    tokens_ids = layers.Input(shape=(None, ), dtype='int32', name='token_input')
    # output shape == (batch_size, seq_len)

    tokens_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size)(tokens_ids)

    biencoder = layers.Bidirectional(layer=layers.LSTM(units=4, return_sequences=True))(tokens_embeddings)

    uniencoder = layers.LSTM(units=4)(biencoder)

    return Model(tokens_ids, uniencoder)

def context_encoder(self, utterance_enc_model):
        context_tokens_ids = layers.Input(
            shape=(3, 30), # context_size, sequence lenght
            )
        # output shape == (batch_size, context_size, seq_len)

        context_utterance_embeddings = layers.TimeDistributed(
            layer=utterance_enc_model, input_shape=(3, 30))(context_tokens_ids)
        # output shape == (batch_size, context_size, utterance_encoding_dim)

        context_encoding = layers.LSTM(units=4, name='context_encoder')(context_utterance_embeddings)
        # output shape == (batch_size, hidden_layer_dim)

        return Model(context_tokens_ids, context_encoding)

            


def decoder():
    pass

def model():
    pass




