import time
from tensorflow.keras.layers import Dense
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
import numpy as np
from tensorflow.python.keras.engine.input_layer import Input
from test import dataset, prediction
from test import encoded
import json
# import tensorflow


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

    # make generative// unsupervised program with another type of learninf ( next sequence for answer and the others (action/ requirements) linear regression)
    emotion_input = Input(shape=emotion_output.shape)
    speech_input = Input(shape=speech_output.shape)

    x = Dense(4, activation="relu")(emotion_input)
    x = Model(inputs=emotion_input, outputs=x)
    # just make a model linear

    y = Dense(4, activation="relu")(speech_input)
    y = Model(inputs=speech_input, outputs=y)

    combined = concatenate([x.output, y.output])

    z = Dense(2, activation="relu")(combined)
    z = Dense(1, activation="linear")(z)

    model = Model(inputs=[x.input, y.input], outputs=z)
    return model, output


def train_normal_block():
    # fit the two model
    model2, x_emotion_train, y_emotion_train, t2 = model_emotion("emotion_pattern.json").model2()
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model2.fit(x_emotion_train, y_emotion_train, epochs=100, batch_size=2, verbose=1)

    time.sleep(10)

    model1, x_speech_train, y_speech_train, t1 = model_speech_rec("ir.json").model1()
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model1.fit(x_speech_train, y_speech_train, epochs=100, batch_size=2, verbose=1)

    return model1, model2


def drop_data(path, emotion_output, speech_output, trained_output):
    data2 = {
        "emotion": emotion_output[0],
        "speech": speech_output[0],
        "output": str(trained_output),
    }
    # add more data and more tags
    with open(f"data/{path}") as file:
        data = json.load(file)

    data["intents"].append(data2)
    file.seek(0)
    json.dump(data, file, indent=4)
    file.close

def get_data_from_file(path):
    with open(f"data/{path}") as file:
        data = json.load(file)

    file.close
    return data["intents"]

def get_training_data():
    model1, model2 = train_normal_block()

    # get a list of pred the test on
    forexample_list = ["i like soup", "turn on the tv"]
    for i in forexample_list:
        pred_1 = model1.predict(forexample_list[i])
        pred_2 = model2.predict(forexample_list[i])
        drop_data("super_block.json", list(pred_2), list(pred_1), forexample_list[i])


def train_super_block():
    lol = get_data_from_file("super_block.json")
    doc1 = []
    doc2 = []
    doc_output = []
    for i in lol[0]:
        output = lol["output"]

        emotion_output = lol["emotion"]

        speech_output = lol["speech"]

        doc1.append(list(speech_output))
        doc2.append(list(emotion_output))
        doc_output.append(str(output))

    doc1 = np.array(doc1)
    doc2 = np.array(doc2)

    # find a way to encode doc_output (generation)