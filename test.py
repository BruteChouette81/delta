from tensorflow.keras.preprocessing.text import Tokenizer
import tensorflow as tf
from tensorflow import keras
import numpy as np
import json
import random
import re


# tokenize the dataset and create a word index
class dataset:

    def __init__(self, json_file):
        self.json_file = json_file

    def open_json(self):
        with open(str(self.json_file)) as file:
            data = json.load(file)

        classes = []
        documents = []
        for intent in data['intents']:
            for pattern in intent['patterns']:
                documents.append((pattern, intent['tag']))

                if intent['tag'] not in classes:
                    classes.append(intent['tag'])
        bag = []
        for words, labels in documents:
            bag.append(words)

        t = Tokenizer()
        t.fit_on_texts(bag)
        return bag, documents, t, classes


# encode data to intergers
class encoded:
    def __init__(self, bag, documents, t, classes):
        self.bag = bag
        self.documents = documents
        self.t = t
        self.classes = classes

    def encoded_doc(self):
        doc = self.t.texts_to_sequences(self.bag)
        p_doc = keras.preprocessing.sequence.pad_sequences(doc, maxlen=5, padding="post")

        out_empty = [0] * len(self.classes)
        training = []

        i = 0

        for sentence, labels in self.documents:
            output = list(out_empty)
            output[self.classes.index(labels)] = 1
            training.append(output)
            i += 1

        return training, p_doc


# get a responce from delta
class responce:
    def __init__(self, path, tag):
        self.path = path
        self.tag = tag

    def get_responce(self):
        with open(str(self.path)) as file:
            data = json.load(file)

        for tg in data["intents"]:
            if tg["tag"] == self.tag:
                responces = tg["responses"]

        random_responce = random.choice(responces)

        return random_responce


# prepare the train data
def prepare_train():
    index = dataset("data/ir.json")
    bag, documents, t, classes = index.open_json()

    encoded_data = encoded(bag, documents, t, classes)
    train = encoded_data.encoded_doc()

    x_train = train[1]
    x_train = np.array(x_train)
    y_train = train[0]
    y_train = np.array(y_train)

    return x_train, y_train, t, classes


# the model
def model():
    x_train, y_train, t, classes = prepare_train()

    model = keras.Sequential([
        keras.layers.Embedding(33, 128),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(len(y_train[0]), activation='sigmoid')
    ])
    model.summary()

    # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # model.fit(x_train, y_train, epochs=100, batch_size=512, verbose=1)
    # return the model

    return model, t, classes


# prediction function
def prediction(inp, model, t):
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

    pred = model.predict(x)
    return pred


# when the programs is called
if __name__ == '__main__':
    model, t, classes = model()
    print(f"model inputs: {model.input}")
    # model.save("model.h5")

    chat = True
    while chat:
        inp = input("enter text: ")
        if inp == 'exit':
            chat = False
        else:
            pred = prediction(inp=inp, model=model, t=t)
            results_index = np.argmax(pred)
            tag = classes[results_index]
            print(pred)

            print(responce("ir.json", tag).get_responce())
