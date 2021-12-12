import keras
import numpy as np

from test import dataset
from test import encoded

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


def train():
    model1, x_speech_train, y_speech_train, t1 = model_speech_rec("ir.json").model1()
    model1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model1.fit(x_speech_train, y_speech_train, epochs=100, batch_size=2, verbose=1)
    model1.save("../server_test/models/model1.h5")