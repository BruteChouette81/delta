import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model

hidden_layer_dim = 64

def encoder(vocab_size, embedding_size):
    tokens_ids = layers.Input(shape=(None, ), dtype='int32', name='token_input')
    # output shape == (batch_size, seq_len)

    tokens_embeddings = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size)(tokens_ids)

    biencoder = layers.Bidirectional(layer=layers.LSTM(units=4, return_sequences=True))(tokens_embeddings)

    uniencoder, forward_h, forward_c = layers.LSTM(units=4, return_state=True)(biencoder)
    encoder_states = [forward_h, forward_c]

    return( Model(tokens_ids, uniencoder), encoder_states)

def context_encoder(utterance_enc_model):
        context_tokens_ids = layers.Input(
            shape=(3, 30), # context_size, sequence lenght 
            )
        # output shape == (batch_size, context_size, seq_len)

        context_utterance_embeddings = layers.TimeDistributed(
            layer=utterance_enc_model, input_shape=(3, 30))(context_tokens_ids)
        # output shape == (batch_size, context_size, utterance_encoding_dim)

        context_encoding, state_h, state_c = layers.LSTM(units=4, name='context_encoder', return_state=True)(context_utterance_embeddings)
        context_states = [state_h, state_c]
        # output shape == (batch_size, hidden_layer_dim)

        return (Model(context_tokens_ids, context_encoding), context_states)


class Decoder(layers.Layer):
    def __init__(self, units, numtokens):
        super(Decoder, self).__init__()
        self.units = units
        self.num_tokens = numtokens
        self.lstm2 = layers.LSTM(self.units, name="decoder")
        self.dense1 = layers.Dense(self.num_tokens, activation='softmax')

    def call(self, inputs, encoders_outputs): #encoders_outputs = Concat([state_h, state_c] + [state_h, state_c])
       

        decoder_outputs = self.lstm2(inputs, initial_state=encoders_outputs)
        decoder_dense = self.dense1(decoder_outputs)

        return decoder_dense


#simple adaptation of a context awarness model that uses two encoder in order to give better and smartest responses in a seq to seq chatbot usage
def model():
    encoder_model, enc_states = encoder(1000, 3)
    print(enc_states)

    context_encoder_model, context_states = context_encoder(encoder_model)
    print(context_states)

    encoder_outputs = layers.concatenate(inputs=[enc_states[0], context_states[0]])
    encoder_outputs2 = layers.concatenate(inputs=[enc_states[1], context_states[1]])
    print(encoder_outputs)

    decoder_inputs = layers.Input(shape=(None,), dtype="int32", name="decoder_inputs")
    decoder_embeddings = layers.Embedding(
            input_dim=1000,
            output_dim=3)(decoder_inputs)

    decoder_dense = Decoder(8, 1000)(decoder_embeddings, [encoder_outputs, encoder_outputs2]) #enc_states

    return Model(inputs=[encoder_model.input, context_encoder_model.input, decoder_inputs], outputs=[decoder_dense])



seq2seq = model()

#model achitecture: 
#   input(text)         input(context)
#   encoder1              encoder2
#          concat(states)
#           decoder
print(seq2seq.summary())

