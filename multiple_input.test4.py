
import re
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling1D
from data.data import get_final_data, index_vocab

from emotions_rec import get_emotion_model, model_emotion, new_model_test
disable_eager_execution()

from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import concatenate
from keras.models import load_model
import numpy as np
from tensorflow.python.keras.engine.input_layer import Input

from speak_rec import get_speech_model, model_speech_rec, new_model_test1

tf.compat.v1.experimental.output_all_intermediates(True)

INMAXLEN = 25
OUTMAXLEN = 27

def process_output():
    # make generative// unsupervised program with another type of learning ( next sequence for answer and the others (action/ requirements) linear regression)

    #emotion_model = new_model_test(3) # freeze the model to pretrain it
    emotion_model = get_emotion_model()
    emotion_model.trainable = False
    emotion_dense = Dense(64, activation = "relu")(emotion_model.output)

    #speech_model = new_model_test1(3
    speech_model = get_speech_model()
    speech_model.trainable = False
    speech_dense = Dense(64, activation = "relu")(speech_model.output)

    #### text model
    text_input = Input(shape=(25, 18)) # modify shape (maxlen, len(vocab_input))
    encoder_lstm = LSTM(64, input_shape=(25, 18), return_sequences=True, return_state=False)(text_input)# change this to dense
    outputs_text = Dense(64, activation="relu")(encoder_lstm)
    text_model = Model(inputs=text_input, outputs=outputs_text)
    
    combined = concatenate(inputs=[tf.reshape(emotion_dense, [-1, 1, 64]), tf.reshape(speech_dense, [-1, 1, 64]), text_model.output], axis=1) # make the two first output 3-dim

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

    model = Model(inputs=[emotion_model.input, speech_model.input, text_input, decoder_input], outputs=decoder_outputs)
    print(model.summary())
    return model


def train_normal_block():
    # fit the two tokenizers
    
    notusedmodel2, x_emotion_train, y_emotion_train, t2 = model_emotion("emotion_pattern.json").model2()
    
    notusedmodel1, x_speech_train, y_speech_train, t1 = model_speech_rec("ir.json").model1()
    
    #load the trained model
    #model2 = load_model("../server_test/models/model2.h5")
    #model1 = load_model("../server_test/models/model1.h5")
    return t1, t2 # model1, model2



doc1, doc2, x_inp_data, x_out_data, inp_training, inp_vocab, out_vocab, out_training, doc_input = get_final_data() # doc1 = speech, doc2 = emotion
# chift one char from the output to make input to lstm decoder and the original will be target

t1, t2 = train_normal_block() #model1, model2, ...

def prep_for_prediction(inp, t):
    # clean and interger encode the inputs
    clean = re.sub(r'[^ a-z A-Z 0-9]', "", inp)
    test_word = clean.split()
    numeric_ls = t.texts_to_sequences(test_word)

    # if the word is not in word index, put 0
    if [] in numeric_ls:
        numeric_ls = list(filter(None, numeric_ls))

    # pad the interger
    numeric_ls = np.array(numeric_ls).reshape(1, len(numeric_ls))
    x = keras.preprocessing.sequence.pad_sequences(numeric_ls, maxlen=5, padding="post")
    return x

x_inp_speech = []
x_inp_emotion = []
for inp in doc_input:
    speech = prep_for_prediction(inp, t1)
    x_inp_speech.append(speech)
    emotion = prep_for_prediction(inp, t2)
    x_inp_emotion.append(emotion)

x_inp_emotion = np.array(x_inp_emotion)
x_inp_emotion = x_inp_emotion.reshape((5,5))

x_inp_speech = np.array(x_inp_speech)
x_inp_speech = x_inp_speech.reshape((5,5))


x_out_targ = []
for i in x_out_data:
    x_out_targ.append(i[1:] + [[0] * 22])

x_inp_data = np.array(x_inp_data)
x_out_data = np.array(x_out_data)
x_out_targ = np.array(x_out_targ)

print(f"X input data shape: {x_inp_emotion.shape}")
### TODO a decision making system 

multi_modal = process_output()

multi_modal.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"])
multi_modal.fit([x_inp_emotion, x_inp_speech, x_inp_data, x_out_data], x_out_targ, epochs=20, batch_size=1)

multi_modal.save("multimodal/model3")

'''
multi_modal = process_output()

multi_modal.load_weights("multimodal/model3")
encoder_input_1 = multi_modal.input[0]  # input_1
encoder_input_2 = multi_modal.input[1]  # input_2
encoder_input_3 = multi_modal.input[2]  # input_3 
#print(multi_modal.layers)
encoder_outputs, state_h_enc, state_c_enc = multi_modal.layers[23].output  # lstm_1 [8]
encoder_states = [state_h_enc, state_c_enc]
encoder_model = Model([encoder_input_1, encoder_input_2, encoder_input_3], encoder_states)

decoder_inputs = multi_modal.input[3]  # input_4
decoder_state_input_h = keras.Input(shape=(64,))
decoder_state_input_c = keras.Input(shape=(64,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_lstm = multi_modal.layers[24] # lstm_2
decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs
)
decoder_states = [state_h_dec, state_c_dec]
decoder_dense = multi_modal.layers[25]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
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
    x_speech = prep_for_prediction(inp, t1)
    inp_speech = x_speech.reshape((1,5))

    x_emotion = prep_for_prediction(inp, t2)
    inp_emotion = x_emotion.reshape((1,5))

    #encode sequence
    seq = index_vocab(inp_vocab, inp_training, inp, True)
    seq = np.array(seq)
    seq = seq.reshape(-1, 25, 18)
    states_value = encoder_model.predict([inp_emotion, inp_speech, seq])
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
'''
input = str(input("type>"))
input = " " + input + "\n"
'''
pred = decode_sentence(" hello\n")
#print(pred)

sentence = token_char(pred, out_vocab)
#print(f"User > {input}")
print(f"Bot > {sentence}")
'''