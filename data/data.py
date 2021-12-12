
import numpy as np
import json

from test import prediction

INMAXLEN = 25
OUTMAXLEN = 27


###############################
###### get the training data###
###############################


#extract all data from a line of text
def extract_data(line):
    if "human:" in line:
        text = line.replace("human:", "")

    else:
        text = line.replace("bot:", "")

    return text

#get the raw data
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

#put trainable data into json file
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

#get prediction from rec model to help the final model
def get_training_data(model1, model2, t1, t2, path):
    #model1, model2, t1, t2 = train_normal_block()

    # get a list of pred the test on
    inp, out = get_data_from_text(path)
    for i in range(len(inp)):
        pred_1 = prediction(inp[i], model1, t1)
        pred_2 = prediction(inp[i], model2, t2)
        drop_data(path, list(pred_2), list(pred_1), inp[i], out[i])


#########################################
###### translating the data to machine ##
#########################################
def create_vocab(output):
    vocab = []
    training = []
    for sample in output:
        for char in sample:
            if char not in vocab:
                vocab.append(char)

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


#########################
## put it all together ##
#########################

def get_data_from_file(path):
    with open(f"data/{path}", "r+") as file:
        data = json.load(file)

    file.close()
    return data["intents"]


def get_final_data():
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
        seq = index_vocab(out_vocab, out_training, sample, False)
        x_out_data.append(seq)

    for sample in doc_input:
        seq = index_vocab(inp_vocab, inp_training, sample, True)
        x_inp_data.append(seq)

    return doc1, doc2, x_inp_data, x_out_data, inp_training, inp_vocab, out_vocab, out_training



def make_new_data(model1, model2, t1, t2):
    #delete everything from the file before
    path = "super_block.json"
    get_training_data(model1, model2, t1, t2, path)