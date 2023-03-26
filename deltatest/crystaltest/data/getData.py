#the goal of this program is to make a training dataset and a vectorizer for the Crystal v1 model
#must create a dataset with: 1: context, 2: text, 3: emotion
# 
# example of one training sequence: 
#   context: "[start] hello, how are you [sep] I'm fine and you ? [sep] Not that good [sep]",
#   text: "what happened ?",
#   emotion: "empathy" 

import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization


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

#encoder/decoder vectorize layers
vectorize_layer_autoenc = TextVectorization(
    max_tokens=40000,
    output_mode="int",
    output_sequence_length=30,
    standardize=preprocess_text
)

def transform_autoenc(inp, out, emo, con): ### add emotion support
    inp, out = tf.expand_dims(inp, -1), tf.expand_dims(out, -1)
    enc, dec = vectorize_layer_autoenc(inp), vectorize_layer_autoenc(out)
    enc = enc[0]
    dec = tf.pad(dec[0], [[0, 1]])
    return (
        {"encoder_inputs": enc, "decoder_inputs": dec[:-1], "emotion_inputs": emo, "condition_inputs": con},
        {"output": dec[1:]},
        
    )

def load_crystal_vectorizer():
    x_data_set = []
    y_data_set = []

    x_condition = []
    y_condition = []

    x_context = []
    #y_context = []


    with open("deltatest/data/generative_data5.txt", "r", encoding = "utf-8", errors="ignore") as file:
        lines = file.readlines()

    max_line = 0
    for line in lines:
        line = line.replace(" â€™ ", " ' ")
        conv = line.split("[sep]")
        x_context_set = [] # reset at each conversation
        if (len(conv) % 2) == 0: # if conversation is even on both size (every iteration as a response)
            count_line = 0
            for text in conv: # iterate over all the text 
                x_context_set[count_line] = x_context[count_line - 1] + text #keep track of the conversation history
                if (count_line % 2) == 0 or count_line == 0: # since each even number a odd number following, separate the text in iteration | response
                    x_data_set.append(str(text))
                else:
                    y_data_set.append(str(text))
                count_line += 1

        else: 
            conv = conv[:-1] # if not even, remove the last iteration.
            count_line = 0
            for text in conv: # iterate over all the text 
                x_context_set[count_line] = x_context[count_line - 1] + text #keep track of the conversation history
                if (count_line % 2) == 0 or count_line == 0: # since each even number a odd number following, separate the text in iteration | response
                    x_data_set.append(str(text))
                else:
                    y_data_set.append(str(text))
                count_line += 1

        if max_line == 300: # top a 300 conversations
            x_context.append(x_context_set) #.toString()
            break

        else:
            max_line+=1
            x_context.append(x_context_set) #.toString()
        

    print(len(x_data_set))
    print(len(y_data_set))


    with open("deltatest/data/emotion.txt", "r", encoding = "utf-8", errors="ignore") as file:
        condition_lines = file.readlines()
    
    max_line = 0
    for line in condition_lines:
        condition_list = line.split(" ") #get all condition for 
        condition_list = condition_list[:-1] #remove "\n"
        
        if (len(condition_list) % 2) == 0:
            count_line = 0
            for condition in condition_list: 
                if (count_line % 2) == 0 or count_line == 0: # since each even number a odd number following, separate the text in iteration | response
                    x_condition.append(int(condition))
                else:
                    y_condition.append(int(condition))

                count_line += 1

        else: 
            condition_list = condition_list[:-1] # if not even, remove the last iteration.
            count_line = 0
            for condition in condition_list: # iterate over all the text 
                if (count_line % 2) == 0 or count_line == 0: # since each even number a odd number following, separate the text in iteration | response
                    x_condition.append(int(condition))
                else:
                    y_condition.append(int(condition))
                count_line += 1

        if max_line == 300: # top a 300 conversations
            break

        else:
            max_line+=1


    print(y_condition[0])
    print(len(y_condition))



    dataset = tf.data.Dataset.from_tensor_slices((x_data_set, y_data_set, y_condition))

    vectorize_layer_autoenc.adapt(tf.data.Dataset.from_tensor_slices((x_data_set + y_data_set)).batch(128)) #batch_size
    
    dataset = dataset.map(transform_autoenc) #num_parallel_calls=tf.data.AUTOTUNE
    train_dataset = (
        dataset.cache()
        .shuffle(20000)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_dataset, vectorize_layer_autoenc