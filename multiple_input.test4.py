
import re
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
from data.data import get_final_data, index_vocab

from emotions_rec import get_emotion_model, model_emotion, new_model_test
#disable_eager_execution()

from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import load_model
import numpy as np
from tensorflow.python.keras.engine.input_layer import Input

from speak_rec import get_speech_model, model_speech_rec, new_model_test1

tf.compat.v1.experimental.output_all_intermediates(True)

INMAXLEN = 25
OUTMAXLEN = 27


#the model 
def process_output(trainable):
    # make generative// unsupervised program with another type of learning ( next sequence for answer and the others (action/ requirements) linear regression)

    #emotion_model = new_model_test(3) # freeze the model to pretrain it
    emotion_model = get_emotion_model(False)
    if not trainable:
        emotion_model.trainable = False

    emotion_dense = Dense(64, activation = "relu")(emotion_model.output) #remove

    #speech_model = new_model_test1(3
    speech_model = get_speech_model(False)
    if not trainable:
        speech_model.trainable = False

    speech_dense = Dense(64, activation = "relu")(speech_model.output) #remove 

    #### text model
    text_input = Input(shape=(25, 18)) # modify shape (maxlen, len(vocab_input))
    encoder_lstm = LSTM(64, input_shape=(25, 18), return_sequences=True, return_state=False)(text_input) # transformer encoder like other (text) modalities
    outputs_text = Dense(64, activation="relu")(encoder_lstm)
    text_model = Model(inputs=text_input, outputs=outputs_text)
    
    combined = concatenate(inputs=[tf.reshape(emotion_dense, [-1, 1, 64]), tf.reshape(speech_dense, [-1, 1, 64]), text_model.output], axis=1) # make the two first output 3-dim

    encoder = LSTM(64, return_state=True)
    
    encoder_outputs, state_h, state_c = encoder(combined)

    encoder_states = [state_h, state_c]


    ### decoder (can change)

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

def getData(x_out_data, x_inp_data, t1, t2):
    # chift one char from the output to make input to lstm decoder and the original will be target

    t1, t2 = train_normal_block() #model1, model2, ... ###change load pre-trained tokenizer 

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

    #shift 1 char for target 
    x_out_targ = []
    for i in x_out_data:
        x_out_targ.append(i[1:] + [[0] * 22])

    x_inp_data = np.array(x_inp_data)
    x_out_data = np.array(x_out_data)
    x_out_targ = np.array(x_out_targ)

    #X target data: x_out_targ[0][1] is equal to x out data: x_out_data[0][2] since at [0][0], the model want to learn to predict the next char.
    return x_inp_emotion, x_inp_speech, x_inp_data, x_out_data, x_out_targ

### TODO a decision making system 

def trainModel3(x_inp_emotion, x_inp_speech, x_inp_data, x_out_targ, epoch: int, batch_size:int, save:bool = True, two_time: bool = False, second_epoch: int = 0, second_batch_size: int = 1):
    multi_modal = process_output(False) # load model (classification non-trainable)
    #fit the model for first time
    multi_modal.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"]) #rmsprop
    multi_modal.fit([x_inp_emotion, x_inp_speech, x_inp_data, x_out_data], x_out_targ, epochs=epoch, batch_size=batch_size)

    if save: 
        multi_modal.save("models_test/model3")
    
    else:
        return multi_modal

    if two_time:
        multi_modal = load_model("models_test/model3")
        multi_modal.trainable = True

        multi_modal.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=["accuracy"]) #rmsprop
        multi_modal.fit([x_inp_emotion, x_inp_speech, x_inp_data, x_out_data], x_out_targ, epochs=second_epoch, batch_size=second_batch_size)
        multi_modal.save("models_test/model3")
        


def buildEncoder(model):
    encoder_input_1 = model.input[0]  # input_1
    encoder_input_2 = model.input[1]  # input_2
    encoder_input_3 = model.input[2]  # input_3 
    #print(model.layers)
    encoder_outputs, state_h_enc, state_c_enc = model.layers[23].output  # lstm_1 [8]
    encoder_states = [state_h_enc, state_c_enc]
    encoder_model = Model([encoder_input_1, encoder_input_2, encoder_input_3], encoder_states)
    return encoder_model

def buildDecoder(model):
    decoder_inputs = model.input[3]  # input_4
    decoder_state_input_h = keras.Input(shape=(64,))
    decoder_state_input_c = keras.Input(shape=(64,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_lstm = model.layers[24] # lstm_2
    decoder_outputs, state_h_dec, state_c_dec = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs
    )
    decoder_states = [state_h_dec, state_c_dec]
    decoder_dense = model.layers[25]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states
    )
    return decoder_model

def buildAutoEncoder(path: str):
   multi_modal = load_model(path)
   encoder_model = buildEncoder(multi_modal)
   decoder_model = buildDecoder(multi_modal)

   return encoder_model, decoder_model



def token_char(tokens, vocab):
    sentence = ""
    for token in tokens:
        if token == vocab.index('\n'):
            return sentence
        else:
            sentence += str(vocab[token])

    return sentence

def decode_sentence(inp, encoder_model, decoder_model, t1, t2):

    #prepare for classification
    x_speech = prep_for_prediction(inp, t1)
    inp_speech = x_speech.reshape((1,5))

    x_emotion = prep_for_prediction(inp, t2)
    inp_emotion = x_emotion.reshape((1,5))

    #encode sequence for seq to seq 
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
    #loop for writting character by character
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

if __name__ == "__main__":
    doc1, doc2, x_inp_data, x_out_data, inp_training, inp_vocab, out_vocab, out_training, doc_input = get_final_data() # doc1 = speech, doc2 = emotion
    t1, t2 = train_normal_block() #model1, model2, ... ###change load pre-trained tokenizer
    x_inp_emotion, x_inp_speech, x_inp_data, x_out_data, x_out_targ = getData(x_out_data, x_inp_data, t1, t2)
    #trainModel3() # put parameters

    input = str(input("type>"))
    input = " " + input

    encoder, decoder = buildAutoEncoder("models_test/model3")

    pred = decode_sentence(" turn on the tv", encoder, decoder, t1, t2)
    #print(pred)

    sentence = token_char(pred, out_vocab)
    print(f"User > {input}")
    print(f"Bot > {sentence}")
