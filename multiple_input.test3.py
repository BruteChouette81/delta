
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
import tensorflow as tf

from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import concatenate
from keras.preprocessing.text import Tokenizer
from keras.models import load_model
import numpy as np
from tensorflow.python.keras.engine.input_layer import Input
from test import dataset, prediction
from test import encoded
import json
# import tensorflow
INMAXLEN = 25
OUTMAXLEN = 27

def prepare_train(data):
    index = dataset(f"data/{data}")
    bag, documents, t, classes = index.open_json()

    encoded_data = encoded(bag, documents, t, classes)
    train = encoded_data.encoded_doc()

    x_train = train[1]
    x_train = np.array(x_train)
    y_train = train[0]
    y_train = np.array(y_train)

    return x_train, y_train, t, classes




def extract_data(line):
    if "human:" in line:
        text = line.replace("human:", "")

    else:
        text = line.replace("bot:", "")

    return text


class model_speech_rec:
    def __init__(self, data):
        self.data = data

    def model1(self):
        x_train, y_train, t, classes = prepare_train(self.data)
        model = keras.Sequential([
            keras.layers.Embedding(33, 128),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(len(y_train[0]), activation='sigmoid')
        ])
        return model, x_train, y_train, t


class model_emotion:
    def __init__(self, data):
        self.data = data

    def model2(self):
        x_train, y_train, t, classes = prepare_train(self.data)
        model = keras.Sequential([
            keras.layers.Embedding(45, 128),
            keras.layers.GlobalAveragePooling1D(),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(len(y_train[0]), activation='sigmoid')
        ])
        return model, x_train, y_train, t


# combine final output of the two previous model


# output must be response


def process_output(output_vocab):

    # make generative// unsupervised program with another type of learning ( next sequence for answer and the others (action/ requirements) linear regression)
    feature_input = Input(shape=(2, 3))
    text_input = Input(shape=(25, 18)) # modify shape (maxlen, len(vocab_input))

    x = Dense(16, activation="relu")(feature_input)
    x = Dense(64, activation="relu")(x)
    
    encoder_lstm = LSTM(64, input_shape=(25, 18), return_sequences=True, return_state=False)(text_input)
    
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
    # fit the two model
    
    model2, x_emotion_train, y_emotion_train, t2 = model_emotion("emotion_pattern.json").model2()
    #model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model2.fit(x_emotion_train, y_emotion_train, epochs=100, batch_size=2, verbose=1)
    #model2.save("../server_test/models/model2.h5")

    #time.sleep(10)
    
    model1, x_speech_train, y_speech_train, t1 = model_speech_rec("ir.json").model1()
    #model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model1.fit(x_speech_train, y_speech_train, epochs=100, batch_size=2, verbose=1)
    #model1.save("../server_test/models/model1.h5")
    model2 = load_model("../server_test/models/model2.h5")
    model1 = load_model("../server_test/models/model1.h5")
    return model1, model2, t1, t2


def create_vocab(output):
    vocab = []
    training = []
    for sample in output:
        for char in sample:
            if char not in vocab:
                vocab.append(char)

    print(vocab)
    for i in range(len(vocab)):
        bag = [0] * len(vocab)

        bag[i] = 1
        training.append(bag)

    return training, vocab


def index_vocab(vocab, training, sentence, inp):
    seq = []
    for word in sentence:
        for char in word:
            if char in vocab:
                ind = vocab.index(char)
                seq.append(training[ind])

            else:
                print(f"Error: Character {char} not in vocab.")
                return
    if inp:
        if len(seq) > INMAXLEN:
            print("Error: Training sequence have a lenght bigger than the max lenght (MAXLEN) change the max lenght or cut the sequence.")
            return

        else:
            for i in range(INMAXLEN - len(seq)):
                seq.append([0] * len(vocab))

    if not inp:
        if len(seq) > OUTMAXLEN:
            print("Error: Training sequence have a lenght bigger than the max lenght (MAXLEN) change the max lenght or cut the sequence.")
            return

        else:
            for i in range(OUTMAXLEN - len(seq)):
                seq.append([0] * len(vocab))

    return seq


def get_data_from_text(path):
    file = open(f"data/{path}", "r")

    string_input = 'human:'
    string_output = 'bot:'

    index = 0

    line_input = []
    line_output = []
  
    for line in file:  
        index += 1
 
        if string_input in line:
          text = extract_data(line)
          line_input.append(text)

        if string_output in line:
          text = extract_data(line)
          line_output.append(text)
                    
    file.close()
    return line_input, line_output

def drop_data(path, emotion_output, speech_output, trained_input, trained_output):
    data2 = {
        "emotion": emotion_output[0].tolist(),
        "speech": speech_output[0].tolist(),
        "input": str(trained_input),
        "output": str(trained_output)
    }
    # add more data and more tags
    with open(f"data/{path}", "r+") as file:
        data = json.load(file)
        data["intents"].append(data2)
        file.seek(0)
        json.dump(data, file, indent=4)

def get_data_from_file(path):
    with open(f"data/{path}", "r+") as file:
        data = json.load(file)

    file.close()
    return data["intents"]

def get_training_data():
    model1, model2, t1, t2 = train_normal_block()

    # get a list of pred the test on
    inp, out = get_data_from_text("super_block_example.txt")
    for i in range(len(inp)):
        pred_1 = prediction(inp[i], model1, t1)
        pred_2 = prediction(inp[i], model2, t2)
        drop_data("super_block.json", list(pred_2), list(pred_1), inp[i], out[i]) # change for  drop_data("super_block.json", list(pred_2), list(pred_1), forexample_list[i], output) where forexample_list is is input of lstm


def train_super_block():
    # get_training_data()
    lol = get_data_from_file("super_block.json")
    doc1 = []
    doc2 = []
    doc_output = []
    doc_input = []
    for i in lol:
        output = i["output"]
        input = i["input"]

        emotion_output = i["emotion"]
        speech_output = i["speech"]

        doc1.append(list(speech_output))
        doc2.append(list(emotion_output))
        doc_output.append(str(output))
        doc_input.append(str(input))

    doc1 = np.array(doc1)
    doc2 = np.array(doc2)
    out_training, out_vocab = create_vocab(doc_output)
    inp_training, inp_vocab = create_vocab(doc_input)

    x_out_data = []
    x_inp_data = []

    for sample in doc_output:
        print(sample)
        seq = index_vocab(out_vocab, out_training, sample, False)
        x_out_data.append(seq)

    for sample in doc_input:
        print(sample)
        seq = index_vocab(inp_vocab, inp_training, sample, True)
        x_inp_data.append(seq)

    return doc1, doc2, x_inp_data, x_out_data, inp_training, inp_vocab, out_vocab, out_training



doc1, doc2, x_inp_data, x_out_data, inp_training, inp_vocab, out_vocab, out_training = train_super_block()
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
    print(combined_pred)

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
        if check != 5:
            check = check + 1

        else:
            writing = False

        target_seq = np.zeros((1, 1, 22))
        target_seq[0, 0, sampled_token_index] = 1
        pred_list.append(sampled_token_index)

        states_value = [h, c]

    return pred_list

pred = decode_sentence(" turn on lights\n")
print(pred)

print(out_vocab)
