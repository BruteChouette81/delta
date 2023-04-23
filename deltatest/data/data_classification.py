

import numpy as np
#tf.compat.v1.disable_eager_execution()
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from tensorflow.keras import layers
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer

from tensorflow.keras.preprocessing.sequence import pad_sequences

from tqdm import tqdm

emotion_dataset = load_dataset("emotion") #tweet with emotion: sadness (0), joy (1), love (2), anger (3), fear (4), surprise (5).

embedding_dim = 64
max_len = 20

### Transformer cells
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



def prepare_data():
    x_tokenizer = Tokenizer()
    bag = []
    labels = []
    classes = [0, 1, 2, 3, 4, 5]
    count = 0
    for sequence in emotion_dataset['train']['text']:
        bag.append(sequence) #already clean
        if count == 10000: #dont get overdata
            break
        else: 
            count +=1

    count = 0
    for sequence in emotion_dataset['train']['label']:
        labels.append(sequence) #already clean
        if count == 10000: #dont get overdata
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

def transformerClassifier(word_index):
    inputs = layers.Input(shape=(max_len,))
    embedding_layer = TokenAndPositionEmbedding(max_len, word_index + 1, embedding_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embedding_dim, 4, 128)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(6, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def train():
    word_index, x_data, y_data, tokeniser = prepare_data()

    print(len(word_index))
    x_train = np.array(x_data)
    y_train = np.array(y_data)
    print(y_train.shape)
    

    #model1 = model(len(word_index))
    model1 = transformerClassifier(len(word_index))
    model1.compile(optimizer='rmsprop', 
              loss='categorical_crossentropy', #change in the future
              metrics=['accuracy'])

    print(model1.summary())
    model1.fit(x_train, y_train, epochs=25, batch_size=64)
    model1.save("categorical_model2")


def test():
    model1 = model(100)
    print(model1.summary())
   

def classify(sentences, tokenizer, model):
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

            if i == 10000: #10,000
                break
            else:
                i+=1

        file.close()
    return convs


if __name__ == '__main__':
    
    #train()
    word_index, x_data, y_data, tokenizer = prepare_data()
    model_pred = keras.models.load_model("./deltatest/data/categorical_model2")
    #pred = classify(["i am very happy to go to denmark"], tokeniser)
    convs=load_dataset()
    #emotionsList = ["sadness", "joy", "love", "anger", "fear", "surprise"]
    #liveEmotion = []

    
    for conv in  tqdm(convs): 

        pred_labels, pred = classify(conv, tokenizer, model_pred)  
        #print(pred)  
        #print("Label detected: " + str(pred_labels))

        with open("deltatest/data/emotion3.txt", mode='a') as file:
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
    
   
