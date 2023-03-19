

import numpy as np
#tf.compat.v1.disable_eager_execution()
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

emotion_dataset = load_dataset("emotion") #tweet with emotion: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).

embedding_dim = 4
max_len = 20

def prepare_data():
    x_tokenizer = Tokenizer()
    bag = []
    labels = []
    classes = [0, 1, 2, 3, 4, 5]
    count = 0
    for sequence in emotion_dataset['train']['text']:
        bag.append(sequence) #already clean
        if count == 1500: #dont get overdata
            break
        else: 
            count +=1

    count = 0
    for sequence in emotion_dataset['train']['label']:
        labels.append(sequence) #already clean
        if count == 1500: #dont get overdata
            break
        else: 
            count +=1

    x_tokenizer.fit_on_texts(bag) #fit the tokenizer to every words in the training dataset
    word_indices = x_tokenizer.texts_to_sequences(bag)
    word_index = x_tokenizer.word_index

    pad_word_indices = pad_sequences(word_indices, max_len, padding='post')

    out_empty = [0] * len(classes)
    training = [] #list of list that contain sparse labels EX: [[0,1,0,0,0,0],[....]]

    for label in labels:
        output = list(out_empty)
        output[classes.index(label)] = 1
        training.append(output)

    return word_index, pad_word_indices, training, x_tokenizer
    

def model(word_index) :
    model = keras.Sequential() 

    model.add(keras.layers.Embedding(word_index + 1, embedding_dim, input_length=max_len)) #basic embedding

    model.add(keras.layers.Conv1D(30,1,activation="relu")) #convolution 1D for better insights
    model.add(keras.layers.MaxPooling1D(4))

    model.add(keras.layers.LSTM(100,return_sequences=True))
    model.add(keras.layers.Flatten()) #to flatten sequences 

    model.add(keras.layers.Dense(500, activation='relu'))
    model.add(keras.layers.Dense(300, activation='relu'))
    #model.add(keras.layers.Dense(50, activation='relu'))
    model.add(keras.layers.Dense(6, activation='softmax')) #num labels
    return model

def train():
    word_index, x_data, y_data, tokeniser = prepare_data()

    print(len(word_index))
    x_train = np.array(x_data)
    y_train = np.array(y_data)
    print(y_train.shape)
    

    model1 = model(len(word_index))
    model1.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', #change in the future
              metrics=['accuracy'])

    model1.fit(x_train, y_train, epochs=60)
    model1.save("categorical_model1")


def test():
    model1 = model(100)
    print(model1.summary())
   

def classify(sentences, tokenizer):
    model = keras.models.load_model("./deltatest/data/categorical_model1")
    pred_labels = []

    word_indices = tokenizer.texts_to_sequences(sentences)
    #print(word_indices)

    pad_word_indices = pad_sequences(word_indices, max_len, padding="post")
    x_data = np.array(pad_word_indices)
    pred = model.predict(x_data)
    for prediction in pred:
        y = np.argmax(prediction)
        pred_labels.append(y)

    return pred_labels, pred


def load_dataset():
    path = "deltatest/data/generative_data5.txt"
    convs = []

    with open(path, encoding = "utf-8", errors="ignore") as file: 
        i=0
        for lines in file:
            lines = lines.replace(" â€™ ", " ' ")
            conv = lines.split("[sep]")
            convs.append(conv)

            if i == 300: 
                break
            else:
                i+=1

        file.close()
    return convs


if __name__ == '__main__':
    #train()
    word_index, x_data, y_data, tokenizer = prepare_data()
    #pred = classify(["i am very happy to go to denmark"], tokeniser)
    convs=load_dataset()
    #emotionsList = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    #liveEmotion = []

    
    for conv in  tqdm(convs): 

        pred_labels, pred = classify(conv, tokenizer)  
        #print(pred)  
        #print("Label detected: " + str(pred_labels))

        with open("deltatest/data/emotion.txt", mode='a') as file:
            labelstr = ''
            for label in pred_labels:
                labelstr += str(label) + ' '
            file.write("\n")
            file.write(labelstr)
            file.close()

        '''
        i = 0
        for label in pred_labels:
            if pred[i][label] > 0.8:
                liveEmotion.append(emotionsList[label])
            else: 
                liveEmotion.append("neutral")
            i+=1

        print(liveEmotion)
        liveEmotion = []
        '''

