
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

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
MAXLEN = 5

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


def process_output(output_vocab, input_vocab):

    # make generative// unsupervised program with another type of learning ( next sequence for answer and the others (action/ requirements) linear regression)
    feature_input = Input(shape=(1, 6))
    text_input = Input(shape=(None, len(input_vocab))) # modify shape

    x = Dense(16, activation="relu")(feature_input)
    x = Dense(8, activation="relu")(x)
    
    encoder_lstm = LSTM(10, input_shape=(1, 3), return_sequences=True, return_state=False)(text_input)
    
    combined = concatenate(inputs=[x, encoder_lstm])
    print(combined.shape)
    '''
    encoder = LSTM(10, return_state=True)
    
    encoder_outputs, state_h, state_c = encoder(combined)

    encoder_states = [state_h, state_c]

    decoder_input = Input(shape=len(output_vocab))
    decoder_lstm = LSTM(10, return_sequences=True, return_state=True)
    decoder_ouput, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
    '''
    
    decoder_lstm = LSTM(10, return_sequences=True, return_state=False)(combined)
    decoder_dense = Dense(len(output_vocab), activation='softmax')(decoder_lstm) #output[0]  for one hot encoding// (decoder_output)
    '''
    z = Dense(16, activation="relu")(combined)
    decoder_dense = Dense(len(output[0]), activation="softmax")(z)
    '''
    model = Model(inputs=[x.input, encoder_lstm.input], outputs=decoder_dense)
    print(model.summary())
    return model


def train_normal_block():
    # fit the two model
    
    model2, x_emotion_train, y_emotion_train, t2 = model_emotion("emotion_pattern.json").model2()
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model2.fit(x_emotion_train, y_emotion_train, epochs=100, batch_size=2, verbose=1)
    #model2.save("../server_test/models/model2.h5")

    #time.sleep(10)
    
    model1, x_speech_train, y_speech_train, t1 = model_speech_rec("ir.json").model1()
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    #model1.fit(x_speech_train, y_speech_train, epochs=100, batch_size=2, verbose=1)
    #model1.save("../server_test/models/model1.h5")
    model2 = load_model("../server_test/models/model2.h5")
    model1 = load_model("../server_test/models/model1.h5")
    return model1, model2, t1, t2

def input_vocab(input):

    input_voc = []
    for sample in input:
        for char in sample:
            if char not in vocab:
               input_voc.append(char)

    return input_voc


def output_vocab(output):
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
          line_input.append(index)

        if string_output in line:
            line_output.append(index)
                    
    print('input at line: ', line_input)
    file.close()

def drop_data(path, emotion_output, speech_output, trained_output):
    data2 = {
        "emotion": emotion_output[0].tolist(),
        "speech": speech_output[0].tolist(),
        "output": str(trained_output),
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
    forexample_list = get_data_from_text("super_block_example.txt")
    for i in range(len(forexample_list)):
        pred_1 = prediction(forexample_list[i], model1, t1)
        pred_2 = prediction(forexample_list[i], model2, t2)
        drop_data("super_block.json", list(pred_2), list(pred_1), forexample_list[i]) # change for  drop_data("super_block.json", list(pred_2), list(pred_1), forexample_list[i], output) where forexample_list is is input of lstm


def train_super_block():
    # get_training_data()
    lol = get_data_from_file("super_block.json")
    doc1 = []
    doc2 = []
    doc_output = []
    for i in lol:
        output = i["output"]

        emotion_output = i["emotion"]
        speech_output = i["speech"]

        doc1.append(list(speech_output))
        doc2.append(list(emotion_output))
        doc_output.append(str(output))

    doc1 = np.array(doc1)
    doc2 = np.array(doc2)
    training, vocab = output_vocab(doc_output)
    
    return doc1, doc2, training, vocab

#get_data_from_text("super_block_example.txt")

doc1, doc2, training, vocab = train_super_block()
print(vocab) # chift one char from the output to make input to lstm decoder and the original will be target
combined_doc = np.column_stack((doc1, doc2))
combined_doc = combined_doc.reshape(-1, 1, 6)
print(combined_doc)
print(training) # we have to encode all data as one hot encoding and then modify the shape of the model ( num_sample, maxlen, vocab_size)

'''
model = process_output(vocab)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit([doc1, doc2], p_bag,
          epochs=5,
          batch_size=1)
'''

'''
inp = input("test: ")
model1, model2, t1, t2 = train_normal_block()

pred_1 = prediction(inp, model1, t1)
pred_1 = list(pred_1)
pred_1 = np.array(pred_1).reshape(-1, 1, 3)

pred_2 = prediction(inp, model2, t2)
pred_2 = list(pred_2)
pred_2 = np.array(pred_2).reshape(-1, 1, 3)
print(pred_2)

pred = model.predict([pred_1, pred_2])
print(pred)
'''