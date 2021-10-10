import json

from transformers import BertTokenizer, BertModel
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


import torch
from torch.utils.data import DataLoader
import torch.nn as nn

from sklearn import preprocessing

import joblib

TOKENIZER = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=True)
MODEL = BertModel.from_pretrained('bert-base-uncased')
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MAX_LEN = 61 + 2
EPOCH = 15
BATCH_SIZE = 32
PATH = "models/net.bin"

class get_data:
    def __init__(self, path_file):
        self.path_file = path_file


    def get_json_data(self):
        with open(self.path_file) as file:
            data = json.load(file)

        classes = []
        Label = []
        ner = []
        documents = []
        doc = []
        for entity in data['entity']:
            for pattern in entity['patterns']:
                documents.append((pattern, entity['tag'], entity['sentence']))
                classes.append(entity['tag'])

        for l in classes:
            for word in l:
                if word not in Label:
                    Label.append(word)

        for sentence, labels, num_sents in documents:
            inp = TOKENIZER.tokenize(sentence)
            max_len = len(inp)
            for i in range(max_len):
                doc.append((inp[i], labels[i]))
            
        return doc, ner



class entity:

    def __init__(self, text, entity, tokenizer, max_len):
        self.text = text
        self.entity = entity
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, item):
        text = self.text[item]
        entity = self.entity[item]

        ids = []
        target_entity = []
        for i, sentence in enumerate(text):

            token_ids = self.tokenizer.encode(sentence, add_special_tokens=False)
            word_piece_entity = [entity[i]] * len(token_ids)
       
            ids.extend(token_ids)
            target_entity.extend(word_piece_entity)

        ids = ids[:self.max_len - 2]
        target_entity = target_entity[:self.max_len - 2]


        ids = [101] + ids + [102]
        target_entity = [0] + target_entity + [0]
        
        
        mask,token_type_id = [1]*len(ids),[0]*len(ids)

        padding_len = MAX_LEN - len(ids)
        
        ids = ids + ([0] * padding_len)
        target_entity = target_entity + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_id = token_type_id + ([0] * padding_len)

        return {
            'ids' : torch.tensor(ids,dtype=torch.long),
            'target_entity' : torch.tensor(target_entity, dtype=torch.long),
            'mask' : torch.tensor(mask, dtype=torch.long),
            'token_type_id' : torch.tensor(token_type_id, dtype=torch.long)
        }

class entity_model(nn.Module):

    def __init__(self, model, num_entity):
        super(entity_model, self).__init__()
        self.model = model
        self.num_entity = num_entity
        #better label linear
        self.drop = nn.Dropout(0.3)
        self.out_entity = nn.Linear(768, self.num_entity)

    def forward(self, ids, mask, token_type_id):
        #add better model (BertForTokenClassification) or my own one
        out = self.model(input_ids = ids, attention_mask = mask, token_type_ids = token_type_id)

        lhs = out['last_hidden_state']

        lhs_entity = self.drop(lhs)
        entity_hs_r = self.out_entity(lhs_entity)

        return entity_hs_r

def entity_loss(logits, targets, mask, num_classes):
    criterion = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_targets = torch.where(
        active_loss,
        targets.view(-1),
        torch.tensor(criterion.ignore_index).type_as(targets)
    )
        
    logits = logits.view(-1, num_classes)

    loss = criterion(logits, active_targets)
    return loss


def transform(ner):

    enc_entity = preprocessing.LabelEncoder()
    entity_transform = enc_entity.fit_transform(ner)

    return entity_transform, enc_entity

def train(model, train_data_loader, optimizer, scheduler):
    final_loss = 0
    model.train()
    for bi, batch in enumerate(train_data_loader):
        for k, v in batch.items():
            batch[k] = v.to(DEVICE)
            
        optimizer.zero_grad()

        out = model(batch['ids'], batch['mask'], batch['token_type_id'])
        loss = entity_loss(out, batch['target_entity'], batch['mask'], model.num_entity)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    final_loss_r = final_loss / len(train_data_loader)
    return final_loss_r


if __name__ == '__main__':

    
    test_text = []
    test_entity = []
    data = get_data("ner.json")
    doc, ner = data.get_json_data()
    for text, label in doc:
        ner.append(label)
        test_text.append(text)

    test, enc_entity = transform(ner=ner)
    meta_data = {
        "enc_tag": enc_entity
    }
    joblib.dump(meta_data, "meta.bin")
    num_entity = len(enc_entity.classes_)
    test = list(test)
    test_text = [test_text]

    #change batch-size
    entity_data = entity(text=test_text, entity=[test], tokenizer=TOKENIZER, max_len=MAX_LEN)

    train_data_loader = DataLoader(entity_data, batch_size=BATCH_SIZE) #  num_workers= 4

    net = entity_model(MODEL, num_entity=num_entity)
    net.to(DEVICE)
    optimizer = AdamW(net.parameters(), lr=2e-5, correct_bias=False)
    #change that
    num_train_steps = EPOCH * BATCH_SIZE #int(60 / EPOCH * BATCH_SIZE)
    scheduler =  get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps = num_train_steps
    )

    for epoch in range(EPOCH):
        print(f"{epoch + 1} on {EPOCH}")
        final_loss_r = train(model = net, train_data_loader = train_data_loader, optimizer = optimizer, scheduler = scheduler)

        print(f"loss: {final_loss_r}")
    torch.save(net.state_dict(), PATH)




