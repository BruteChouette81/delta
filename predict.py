
from entity import entity_model
from entity import entity
from entity import MODEL
from entity import TOKENIZER
from entity import MAX_LEN
from entity import DEVICE

from test import prediction
from test import model

import joblib
import torch
import numpy as np
from keras.models import load_model
import re

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    query: str


#load data
meta_data = joblib.load("models/meta.bin")
enc_tag = meta_data["enc_tag"]
num_tag = len(enc_tag.classes_)


@app.post("/query")
def query(item : Item):

    sentence = item.query
    clean = re.sub(r'[^ a-z A-Z 0-9]', "", sentence)

    sentence_2 = clean.split()
    tk_sentence = TOKENIZER.encode(sentence_2)
    test_data = entity(text=[sentence_2], entity=[[0] * len(sentence_2)], tokenizer=TOKENIZER, max_len=MAX_LEN)

    #intent predict
    t, classes = model()
    intent_ai = load_model("models/model.h5")
    pred = prediction(inp=sentence, model=intent_ai, t=t)
    results_index = np.argmax(pred)
    tag = classes[results_index]

    #print(f"sentence to predict: {sentence_2}")
    #print(f"tag of the sentence: {tag}")

    entity_ai = entity_model(MODEL, num_entity=num_tag)
    entity_ai.load_state_dict(torch.load("models/net.bin"))
    entity_ai.to(DEVICE)

    #ner predict

    def entity_logist(logist):
        logist = logist.view(-1, logist.shape[-1]).cpu().detach()
        probs = torch.softmax(logist, dim=1)
        y_hat = torch.argmax(probs, dim=1)
        return probs.numpy(), y_hat.numpy()

    with torch.no_grad():
        data = test_data[0]
        for k, v in data.items():
            data[k] = v.to(DEVICE).unsqueeze(0)

        tag2 = entity_ai(data['ids'], data['mask'], data['token_type_id'])
        score, pred = entity_logist(tag2)
        score = score[1:len(tk_sentence)-1:, :]
        labels = enc_tag.inverse_transform(pred)[1:len(tk_sentence)-1]

        #print(f"words labels: {labels}")
        #print(score)
    
    return {"intent": str(tag), "NER": str(labels)}






'''
tk_sentence = TOKENIZER.encode(sentence)
sentence = sentence.split()

test_data = entity(text=[sentence], entity=[[0] * len(sentence)], tokenizer=TOKENIZER, max_len=MAX_LEN)
model = entity_model(MODEL, num_entity=num_tag)
model.load_state_dict(torch.load("net.bin"))
model.to(DEVICE)


def entity_logist(logist):
    logist = logist.view(-1, logist.shape[-1]).cpu().detach()
    probs = torch.softmax(logist, dim=1)
    y_hat = torch.argmax(probs, dim=1)
    return probs.numpy(), y_hat.numpy()

with torch.no_grad():
    data = test_data[0]
    for k, v in data.items():
        data[k] = v.to(DEVICE).unsqueeze(0)

    tag = model(data['ids'], data['mask'], data['token_type_id'])
    score, pred = entity_logist(tag)
    score = score[1:len(tk_sentence)-1:, :]
    labels = enc_tag.inverse_transform(pred)[1:len(tk_sentence)-1]

    print(labels)
    print(sentence)
    print(score)






inputs = TOKENIZER.encode_plus(sentence, None, add_special_tokens = True, truncation = True, max_length = MAX_LEN)

ids = inputs['input_ids']
mask = inputs['attention_mask']
token_type_ids = inputs['token_type_ids']
padding_len = MAX_LEN - len(ids)

pad_ids = ids + ([0] * padding_len)
mask = mask + ([0] * padding_len)
token_type_ids = token_type_ids + ([0] * padding_len)

ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0).to(DEVICE)
token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0).to(DEVICE)
'''
