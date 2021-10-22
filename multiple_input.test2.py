
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


def process_output(emotion_output, speech_output, output):

    # make generative// unsupervised program with another type of learning ( next sequence for answer and the others (action/ requirements) linear regression)
    emotion_input = Input(shape=(2, 3))
    speech_input = Input(shape=(2, 3))

    x = Dense(16, activation="relu")(emotion_input)
    x = Dense(8, activation="relu")(x)
    x = Model(inputs=emotion_input, outputs=x)
    # just make a model linear

    y = Dense(16, activation="relu")(speech_input)
    y = Dense(8, activation="relu")(y)
    y = Model(inputs=speech_input, outputs=y)

    combined = concatenate(inputs=[x.output, y.output])
    print(combined.shape)
    
    decoder_lstm = LSTM(10, input_shape=(2, 3), return_sequences=True, return_state=False)(combined)
    decoder_dense = Dense(len(output[0][0]), activation='softmax')(decoder_lstm) #output[0]  for one hot encoding
    '''
    z = Dense(16, activation="relu")(combined)
    decoder_dense = Dense(len(output[0]), activation="softmax")(z)
    '''
    model = Model(inputs=[x.input, y.input], outputs=decoder_dense)
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
    with open(f"data/{path}", "r") as file:
        data = json.load(file)

    file.close
    return data["intents"]

def get_training_data():
    model1, model2, t1, t2 = train_normal_block()

    # get a list of pred the test on
    forexample_list = ["i like soup", "turn on the tv"]
    for i in range(len(forexample_list)):
        pred_1 = prediction(forexample_list[i], model1, t1)
        pred_2 = prediction(forexample_list[i], model2, t2)
        drop_data("super_block.json", list(pred_2), list(pred_1), forexample_list[i])


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
    t = Tokenizer()
    t.fit_on_texts(doc_output)
    # find a way to encode doc_output (generation)
    bag = t.texts_to_sequences(doc_output)
    p_bag = keras.preprocessing.sequence.pad_sequences(bag, maxlen=MAXLEN, padding="post")
    return doc1, doc2, p_bag

doc1, doc2, p_bag = train_super_block()
doc1 = doc1.reshape(-1, 2, 3)
print(doc1)
doc2 = doc2.reshape(-1, 2, 3)
print(doc2)
p_bag = p_bag.reshape(-1, 2, 5)
print(p_bag)

model = process_output(doc2, doc1, p_bag)
model.compile(optimizer='rmsprop', loss='binary_crossentropy')
model.fit([doc1, doc2], p_bag,
          epochs=5,
          batch_size=1)


'''
{
    "emotion": [0.2, 0.01, 0.98],
    "speech": [0.12, 0.92, 0.3],
    "output": "ok boss",
    "_comment": "add more specifics tags"
  }
  {"intents": [

]
}
'''