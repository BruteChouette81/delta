
import re
from this import d
import time
from click import prompt
from datasets import load_dataset
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow import keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import json
import string

conv_ai_dataset = load_dataset("conv_ai_2") #https://raw.githubusercontent.com/alexa/Topical-Chat/master/conversations/train.json
empat_dataset = load_dataset("empathetic_dialogues")
#https://huggingface.co/datasets/gem/viewer/schema_guided_dialog/train

strip_chars = string.punctuation
strip_chars = strip_chars.replace("[", "")
strip_chars = strip_chars.replace("]", "")
def custom_standardization(input_string):
    lowercased = tf.strings.lower(input_string)
    return tf.strings.regex_replace(lowercased, "[%s]" % re.escape(strip_chars), "")

def preprocess_text(sentence):
    sentence = tf.strings.lower(sentence)
    # Adding a space between the punctuation and the last word to allow better tokenization
    sentence = tf.strings.regex_replace(sentence, r"([?.!,])", r" \1 ")
    # Replacing multiple continuous spaces with a single space
    sentence = tf.strings.regex_replace(sentence, r"\s\s+", " ")
    # Replacing non english words with spaces
    sentence = tf.strings.regex_replace(sentence, r"[^a-z?.!,]+", " ")
    sentence = tf.strings.strip(sentence)
    sentence = tf.strings.join(["[start]", sentence, "[end]"], separator=" ")
    return sentence

batch_size = 64 #change to 128
vectorize_layer_generative = TextVectorization(
    max_tokens=40000 - 1,
    output_mode="int",
    output_sequence_length=80 + 1,
    standardize=custom_standardization
)

#encoder/decoder vectorize layers
vectorize_layer_autoenc = TextVectorization(
    max_tokens=10000,
    output_mode="int",
    output_sequence_length=30,
    standardize=preprocess_text
)



def process_x_dataset_emotion(dataset):
    bag = [] #[]
    num_sentence = 0
    for sequence in dataset['train']['dialog']:
        bag.append(sequence)


        if num_sentence == 4:
            print("stopping the loop")
            break

        else:
            num_sentence += 1

    #print(bag[0])
    with open("data.json", "w") as fp:
        json.dump(bag, fp, indent=6)
        fp.close()
    #print(bag)
    '''
    t = Tokenizer()
    t.fit_on_texts(bag)

    doc = t.texts_to_sequences(bag)
    p_doc = keras.preprocessing.sequence.pad_sequences(doc, maxlen=25, padding="post")
    vocab_size = len(t.word_index) + 1
    return p_doc, vocab_size, t
    '''
def load_seq():
    with open("data.json") as fp:
        data = json.load(fp)

    x_data_set = []
    y_data_set = []

    for conversation in data["dialogues"]:
        if len(conversation) < 2:
            continue
        for conv in conversation:
            if conv["id"] % 2 == 0:
                x_data_set.append(conv["text"])

            else:
                y_data_set.append("[START] " + conv["text"] + " [END]")
    y_targ_set = []
    for sentence in y_data_set:
        y_targ_set.append(sentence[7:])

    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(x_data_set)

    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(y_data_set)

    x_doc = x_tokenizer.texts_to_sequences(x_data_set)
    x_pdoc = keras.preprocessing.sequence.pad_sequences(x_doc, maxlen=25, padding="post")
    x_vocab_size = len(x_tokenizer.word_index)

    y_doc = y_tokenizer.texts_to_sequences(y_data_set)
    y_pdoc = keras.preprocessing.sequence.pad_sequences(y_doc, maxlen=25, padding="post")
    y_target = y_tokenizer.texts_to_sequences(y_targ_set)
    y_traget_doc = keras.preprocessing.sequence.pad_sequences(y_target, maxlen=25, padding="post")
    y_vocab_size = len(y_tokenizer.word_index) 
    return x_pdoc, x_vocab_size, x_tokenizer, y_pdoc, y_vocab_size, y_tokenizer, y_traget_doc

#process_x_dataset_emotion(dataset)

### try to put (as X ): [start] context [sep or starthuman] text [end] --> [startbot] responce [end] 
### put mask 
def test_load_vectorization():
    with open("data.json") as fp:
        data = json.load(fp)

    x_data_set = []
    y_data_set = []

    for conversation in data["dialogues"]:
        if len(conversation) < 2:
            continue
        for conv in conversation:
            if conv["id"] % 2 == 0:
                x_data_set.append(conv["text"])

            else:
                y_data_set.append("[START] " + conv["text"] + " [END]")
    
    x_vectorization = TextVectorization(
        max_tokens=50, output_mode="int", output_sequence_length=25,
    )
    x_vectorization.adapt(x_data_set)
    x = x_vectorization(x_data_set)

    y_vectorization = TextVectorization(
        max_tokens=50, output_mode="int", output_sequence_length=25,
    )
    y_vectorization.adapt(y_data_set)
    y = x_vectorization(y_data_set)

    return ({"encoder_inputs": x, "decoder_inputs": y[:, :-1],}, y[:, 1:])

def clean_seq(inp: str):
    inp = inp.lower()
    inp = inp.replace("\n", " ")
    inp = inp.replace("_comma_", ", ")
    emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)

    clean_inp = emoji_pattern.sub(r'', inp)
    return clean_inp


def write_conv_ai():
    bag = []
    illegal = "searching for peer"
    dialogue = ""
    ### the technique here is to combined all sentences of a dialogue
    # into 1 string and adding start and end special tokens
    print("[INFO] start extracting")
    for sequence in conv_ai_dataset['train']['dialog']:
        for text in sequence:
            if len(sequence) > 1: # if it is a real dialogue
                dialogue += str(text['text']) + " "
            else:
                continue
        
        dialogue = "<start> " + dialogue + "<end>"
        dialogue = clean_seq(dialogue)
        bag.append(dialogue)
        dialogue = ""


    print("[INFO] stopping the extraction")
    print("[INFO] start writing")
    with open("generative_data3.txt", "w", encoding = "utf-8", errors="ignore") as fp:
        for text in bag:
            if text != "<start> <end>" and not illegal in text:
                fp.write(text + "\n")
            else:
                continue
        fp.close()

        print("[INFO] stop writing")

def write_conv_ai_autoenc():
    bag_x = []
    bag_y = []
    clean_x = []
    clean_y = []
    illegal = "searching for peer"
    ### the technique here is to combined all sentences of a dialogue
    # into 1 string and adding start and end special tokens
    print("[INFO] start extracting")
    for sequence in conv_ai_dataset['train']['dialog']:
        for text in sequence: 
            if len(sequence) < 2:
                continue
            if (text["id"] % 2) == 0:
                bag_x.append(str(text["text"]))

            else:
                bag_y.append(str(text["text"]))

        
    for dialogue in bag_x:
        clean_dialogue = clean_seq(str(dialogue))
        clean_x.append(clean_dialogue)

    for dialogue in bag_y:
        clean_dialogue = clean_seq(str(dialogue))
        clean_y.append(clean_dialogue)

    bag = list(zip(clean_x, clean_y))


    print("[INFO] stopping the extraction")
    print("[INFO] start writing")
    with open("autoenc_data3.txt", "w", encoding = "utf-8", errors="ignore") as fp:
        for text_x, text_y in bag:
            if not illegal in text_x and not illegal in text_y and text_y != "" and text_x != "":
                fp.write(text_x + "\n")
                fp.write(text_y + "\n")
            else:
                continue
        fp.close()

        print("[INFO] stop writing")

def write_empat():
    bag = []
    dialogues = ""
    print("[INFO] start extracting")
    for sequence in empat_dataset['train']:
        dialogues += "<start> " + str(sequence['prompt']) + " "
        dialogues += str(sequence['utterance']) + " <end>"

        dialogues = clean_seq(dialogues)
        bag.append(dialogues)
        dialogues = ""

    print("[INFO] stopping the extraction")
    print("[INFO] start writing")
    with open("data/empathetic_dialogues.txt", "w", encoding = "utf-8", errors="ignore") as fp:
        for text in bag:
            if text:
                fp.write(text + "\n")
            else:
                continue
        fp.close()

        print("[INFO] stop writing")

def transform_text(text):
    text = tf.expand_dims(text, -1)
    pdoc = vectorize_layer_generative(text)
    px = pdoc[:, :-1]
    py = pdoc[:, 1:]
    return px, py

def load_generative(data_set):
    vectorize_layer_generative.adapt(data_set)
    

    text_ds = data_set.map(transform_text)
    text_ds = text_ds.prefetch(tf.data.AUTOTUNE)
    vocab = vectorize_layer_generative.get_vocabulary()
    

    return vocab, text_ds

def transform_autoenc(inp, out):
    inp, out = tf.expand_dims(inp, -1), tf.expand_dims(out, -1)
    enc, dec = vectorize_layer_autoenc(inp), vectorize_layer_autoenc(out)
    enc = enc[0]
    dec = tf.pad(dec[0], [[0, 1]])
    return (
        {"encoder_inputs": enc, "decoder_inputs": dec[:-1]},
        {"output": dec[1:]},
        
    )

def load_autoenc_vectorizer():
    x_data_set = []
    y_data_set = []
    with open("autoenc_data3.txt", "r", encoding = "utf-8", errors="ignore") as file:
        lines = file.readlines()

    count_line = 0
    for line in lines:
        if (count_line % 2) == 0 or count_line == 0:
            x_data_set.append(str(line))

        else:
            y_data_set.append(str(line))

        count_line += 1

    print(x_data_set[0])
    print(y_data_set[0])

    dataset = tf.data.Dataset.from_tensor_slices((x_data_set, y_data_set))

    vectorize_layer_autoenc.adapt(tf.data.Dataset.from_tensor_slices((x_data_set + y_data_set)).batch(128)) #batch_size
    
    dataset = dataset.map(transform_autoenc) #num_parallel_calls=tf.data.AUTOTUNE
    train_dataset = (
        dataset.cache()
        .shuffle(20000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, vectorize_layer_autoenc

def load_autoenc_tokenizer():
    x_data_set = []
    y_data_set = []

    with open("autoenc_data1.txt", "r", encoding = "utf-8", errors="ignore") as file:
        lines = file.readlines()

    count_line = 0
    for line in lines:
        if count_line % 2 == 0 or count_line == 0:
            x_data_set.append(str(line))

        else:
            y_data_set.append(str(line))

        count_line += 1
    y_targ_set = []
    for sentence in y_data_set:
        y_targ_set.append(sentence[14:])

    x_tokenizer = Tokenizer()
    x_tokenizer.fit_on_texts(x_data_set)

    y_tokenizer = Tokenizer()
    y_tokenizer.fit_on_texts(y_data_set)

    x_doc = x_tokenizer.texts_to_sequences(x_data_set)
    x_pdoc = keras.preprocessing.sequence.pad_sequences(x_doc, maxlen=30, padding="post")
    x_pdoc = np.array(x_pdoc)
    x_vocab_size = len(x_tokenizer.word_index)

    y_doc = y_tokenizer.texts_to_sequences(y_data_set)
    y_pdoc = keras.preprocessing.sequence.pad_sequences(y_doc, maxlen=30, padding="post")
    y_pdoc = np.array(y_pdoc)

    y_target = y_tokenizer.texts_to_sequences(y_targ_set)
    y_target_doc = keras.preprocessing.sequence.pad_sequences(y_target, maxlen=30, padding="post")
    y_target_doc = np.array(y_target_doc)

    y_vocab_size = len(y_tokenizer.word_index) 
    return x_pdoc, x_vocab_size, x_tokenizer, y_pdoc, y_vocab_size, y_tokenizer, y_target_doc


if __name__ == '__main__':
    #write_conv_ai()
    #time.sleep(5)
    #write_empat()
    ### need to put [BOTStart], [HUMANStart] and [END] token
    #write_conv_ai_autoenc()
    
    train_ds, doodoo = load_autoenc_vectorizer()
    for inputs, targets in train_ds.take(1):
        print(f'inputs["encoder_inputs"].shape: {inputs["encoder_inputs"].shape}')
        print(f'inputs["decoder_inputs"].shape: {inputs["decoder_inputs"].shape}')
        print(f"targets['output'].shape: {targets['output'].shape}")

    vocab = vectorize_layer_autoenc.get_vocabulary()
    print(len(vocab))


    #filename = ["data/generative_data3.txt", "data/empathetic_dialogues.txt"]
    #text_ds = tf.data.TextLineDataset(filename) #.filter(lambda x: tf.cast(tf.strings.length(x), bool))
    #text_ds = text_ds.shuffle(buffer_size=256)
    #text_ds = text_ds.batch(batch_size)
    #vocab, text = load_generative(text_ds)
    #print(len(vocab)) # C:\Users\hbari\AppData\Local\Programs\Python\Python39\Lib\site-packages\tensorflow\python\util\compat.py
    #test_prompt = "<start> hello! how is your day?" # [25, 12, 35, 22, 13, 4, 5, 26, 12]
    #text = tf.expand_dims(test_prompt, -1)
    #pdoc = vectorize_layer(text)
    #px = pdoc[:, :-1]
    #print(px) 

    #ds = text.take(1)
    #ds = list(ds.as_numpy_iterator())
    #print(ds)
    #print(f"text data_set shape : {text_ds}")
    #print(f"text dataset type {type(text_ds)}")