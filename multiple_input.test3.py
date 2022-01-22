
from tensorflow.python.framework.ops import disable_eager_execution
from data.data import get_final_data, index_vocab

from emotions_rec import model_emotion
disable_eager_execution()

from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import concatenate
from keras.models import load_model
import numpy as np
from tensorflow.python.keras.engine.input_layer import Input
from test import prediction

from speak_rec import model_speech_rec

INMAXLEN = 25
OUTMAXLEN = 27

def process_output(output_vocab):
    
    # make generative// unsupervised program with another type of learning ( next sequence for answer and the others (action/ requirements) linear regression)
    feature_input = Input(shape=(2, 3)) # put one input per features
    text_input = Input(shape=(25, 18)) # modify shape (maxlen, len(vocab_input))

    x = Dense(16, activation="relu")(feature_input)
    x = Dense(64, activation="relu")(x)
    
    encoder_lstm = LSTM(64, input_shape=(25, 18), return_sequences=True, return_state=False)(text_input)# change this to dense 
    
    combined = concatenate(inputs=[x, encoder_lstm], axis=1) #axis= 1

    encoder = LSTM(64, return_state=True)
    
    encoder_outputs, state_h, state_c = encoder(combined)

    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=(None, 22), name="decoder_input")
    decoder_lstm = LSTM(64, return_sequences=True, return_state=True, name="decoder_lstm")
    decoder_output, _, _ = decoder_lstm(decoder_input,
                                     initial_state=encoder_states)
    #decoder_lstm = LSTM(10, return_sequences=True, return_state=False)(combined)
    decoder_dense = Dense(22, activation='softmax')
    decoder_outputs = decoder_dense(decoder_output) #output[0]  for one hot encoding// (decoder_output)

    model = Model(inputs=[feature_input, text_input, decoder_input], outputs=decoder_outputs)
    print(model.summary())
    return model

def train_normal_block():
    # fit the two tokenizers
    
    notusedmodel2, x_emotion_train, y_emotion_train, t2 = model_emotion("emotion_pattern.json").model2()
    
    notusedmodel1, x_speech_train, y_speech_train, t1 = model_speech_rec("ir.json").model1()
    
    #load the trained model
    model2 = load_model("../server_test/models/model2.h5")
    model1 = load_model("../server_test/models/model1.h5")
    return model1, model2, t1, t2



doc1, doc2, x_inp_data, x_out_data, inp_training, inp_vocab, out_vocab, out_training, notutile = get_final_data()
# chift one char from the output to make input to lstm decoder and the original will be target
combined_doc = np.column_stack((doc1, doc2))
combined_doc = combined_doc.reshape(-1, 2, 3)
print(f"Combined feature: {combined_doc}")
x_out_targ = []
for i in x_out_data:
    x_out_targ.append(i[1:] + [[0] * 22])

x_inp_data = np.array(x_inp_data)
x_out_data = np.array(x_out_data)
x_out_targ = np.array(x_out_targ)

print(f"X input data shape: {x_out_targ.shape}")
### TODO a decision making system 
'''
multi_modal = process_output(x_out_data[0][0])

multi_modal.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
multi_modal.fit([combined_doc, x_inp_data, x_out_data], x_out_targ,
          epochs=100,
          batch_size=1)

multi_modal.save("multimodalities")
'''
multi_modal = load_model("multimodalities")
encoder_input_1 = multi_modal.input[0]  # input_1
encoder_input_2 = multi_modal.input[1]  # input_2
encoder_outputs, state_h_enc, state_c_enc = multi_modal.layers[7].output  # lstm_1 [8]
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model([encoder_input_1, encoder_input_2], encoder_states)

decoder_inputs = multi_modal.input[2]  # input_3
decoder_state_input_h = keras.Input(shape=(64,))
decoder_state_input_c = keras.Input(shape=(64,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = multi_modal.layers[8] # lstm_2 
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = multi_modal.layers[9]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = keras.Model(
    [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
)
def token_char(tokens, vocab):
    sentence = ""
    for token in tokens:
        if token == vocab.index('\n'):
            return sentence
        else:
            sentence += str(vocab[token])

    return sentence

def decode_sentence(inp):
    model1, model2, t1, t2 = train_normal_block()
    #make speech pred.
    pred_1 = prediction(inp, model1, t1)
    pred_1 = list(pred_1)
    
    #make emotion simuli pred.
    pred_2 = prediction(inp, model2, t2)
    pred_2 = list(pred_2)
    
    #combined the signal
    combined_pred = np.column_stack((pred_1, pred_2))
    combined_pred = combined_pred.reshape(-1, 2, 3)
    #print(combined_pred)

    #encode sequence
    seq = index_vocab(inp_vocab, inp_training, inp, True)
    seq = np.array(seq)
    seq = seq.reshape(-1, 25, 18)
    states_value = encoder_model.predict([combined_pred, seq])
    #make a starting point for the decoder
    target_seq = np.zeros((1, 1, 22))
    target_seq[0, 0, 0] = 1 # [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    writing = True
    check = 0
    pred_list = []
    while writing:
        pred, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(pred[0, -1, :])
        if check != 10:
            check = check + 1

        else:
            writing = False

        target_seq = np.zeros((1, 1, 22))
        target_seq[0, 0, sampled_token_index] = 1
        pred_list.append(sampled_token_index)

        states_value = [h, c]

    return pred_list

input = str(input("type>"))
input = " " + input + "\n"
pred = decode_sentence(input)
#print(pred)

sentence = token_char(pred, out_vocab)
#print(f"User > {input}")
print(f"Bot > {sentence}")
