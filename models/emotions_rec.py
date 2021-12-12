import keras 
import numpy as np

from test import encoded
from test import dataset

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



def train():
    model2, x_emotion_train, y_emotion_train, t2 = model_emotion("emotion_pattern.json").model2() # make dataset a parameter
    model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy']) #make loss and opt. params
    model2.fit(x_emotion_train, y_emotion_train, epochs=100, batch_size=2, verbose=1) # amke epoch and batch params
    model2.save("../server_test/models/model2.h5")