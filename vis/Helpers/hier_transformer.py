#!/usr/bin/env python
# coding: utf-8

import transformers
from transformers import BertTokenizerFast
import numpy as np
import re
import string
from nltk.corpus import stopwords
from tensorflow.keras import models, layers
import pickle as pk

def init():

    stop_words = stopwords.words('english')
    max_len = 200   # maxium length of sentence for heirarchical transformer

    id_to_label = {0 : 'Computer', 1: 'Recreation', 2 : 'Science', 3: 'Politics', 4: 'RFP', 5: 'Water Regulations', 6 : 'Person'} # mapping from id to domain

    # Loading BERT tokenizer to convert training data into token ids and input masks for input to heirarchical transformer
    tokenizer = BertTokenizerFast.from_pretrained('distilbert-base-uncased', do_lower_case=True, padding = True, truncation = True)

    ## Loading pretrained 'distilbert-base' model from transformers
    config = transformers.DistilBertConfig(dropout=0.2, 
                attention_dropout=0.2)
    config.output_hidden_states = False
    bert_model = transformers.TFDistilBertModel.from_pretrained('distilbert-base-uncased', config=config)

    return stop_words, max_len, id_to_label, tokenizer, bert_model

stop_words, max_len, id_to_label, tokenizer, bert_model = init()

# Preprocessing method for text data like eliminating numeric, alphanumeric data as well as removing stop words.
def preproc(t):
    
    re_url = re.compile(r'(?:http|ftp|https)://(?:[\w_-]+(?:(?:\.[\w_-]+)+))(?:[\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?')
    re_email = re.compile('(?:[a-z0-9!#$%&\'*+/=?^_`{|}~-]+(?:\.[a-z0-9!#$%&\'*+/=?^_`{|}~-]+)*|"(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21\x23-\x5b\x5d-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])*")@(?:(?:[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.)+[a-z0-9](?:[a-z0-9-]*[a-z0-9])?|\[(?:(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9]))\.){3}(?:(2(5[0-5]|[0-4][0-9])|1[0-9][0-9]|[1-9]?[0-9])|[a-z0-9-]*[a-z0-9]:(?:[\x01-\x08\x0b\x0c\x0e-\x1f\x21-\x5a\x53-\x7f]|\\[\x01-\x09\x0b\x0c\x0e-\x7f])+)\])')
    text = re.sub(r'(From:\s+[^\n]+\n)', '', t)
    text = re.sub(r'(Subject:[^\n]+\n)', '', text)
    text = re.sub(r'(([\sA-Za-z0-9\-]+)?[A|a]rchive-name:[^\n]+\n)', '', text)
    text = re.sub(r'(Last-modified:[^\n]+\n)', '', text)
    text = re.sub(r'(Version:[^\n]+\n)', '', text)
    text = text.lower()
    text = text.strip()
    text = re.sub(re_url, '', text)
    text = re.sub(re_email, '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'(\d+)', ' ', text)
    text = re.sub(r'(\s+)', ' ', text)
    a = ''
    for word in text.split():
        if word not in stop_words and len(word) > 2 and word.isalpha():
            a += word + ' '
    return a

# Converting training data texts into 200 word texts with 50 word overlap (implementation of heirarchical transformer)
def get_split(text1):
    
    l_total = []
    l_parcial = []
    if len(text1.split())//150 >0:
        n = len(text1.split())//150
    else: 
        n = 1
    for w in range(n):
        if w == 0:
            l_parcial = text1.split()[:200]
            l_total.append(" ".join(l_parcial))
        else:
            l_parcial = text1.split()[w*150:w*150 + 200]
            l_total.append(" ".join(l_parcial))
    return l_total

# Loading saved training data
def load_train_data():
    
    op = []
    basepath = r'./data/'
    load_list = ['X_train', 'X_test', 'Y_train', 'Y_test']
    for data in load_list:
        temp = []
        path = basepath + data + '.pickle'
        f = open(path, 'rb')
        temp = pk.load(f)
        f.close()
        op.append(temp)
    
    return op[0], op[1], op[2], op[3]
def get_data(for_transformer = False):

    X_train, X_test, Y_train, Y_test = load_train_data()
    Y_train = np.asarray(Y_train)
    Y_test = np.asarray(Y_test)
    
    if not for_transformer:
        return X_train, X_test, Y_train, Y_test
    
    else:    # Convering training data to tokenized form for heirarchical transformer
        X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=max_len)
        X_test_tokenized = tokenizer(X_test, padding=True, truncation=True, max_length=max_len)
        X_train_input = X_train_tokenized['input_ids']
        X_train_masks = X_train_tokenized['attention_mask']
        X_test_input = X_test_tokenized['input_ids']
        X_test_masks = X_test_tokenized['attention_mask']
        X_train_final = [np.asarray(X_train_input), np.asarray(X_train_masks)]
        X_test_final = [np.asarray(X_test_input), np.asarray(X_test_masks)]

        return X_train_final, X_test_final, Y_train, Y_test

# Creating the Heirarchical Transfomer (BERT - LSTM structure)
def create_model():
    
    idx = layers.Input((max_len), dtype="int32", name="input_idx")
    masks = layers.Input((max_len), dtype="int32", name="input_masks")
    bert_out = bert_model(idx, attention_mask=masks)[0]
    l_mask = layers.Masking(mask_value=-99.)(bert_out)
    encoded_text = layers.LSTM(100, input_shape = (None, 768))(l_mask)  
    out_dense = layers.Dense(30, activation='relu')(encoded_text)
    y_out = layers.Dense(6, activation='softmax')(out_dense)
    
    model = models.Model([idx, masks], y_out)
    
    for layer in model.layers[:3]:
        layer.trainable = False
        
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Loading pretrained weights into the model (By default v7 is being loaded, but user can load any of the available weights in the model_weights folder)
def config_model(path):

    model = create_model()
    model.load_weights(path+'/vis/Data/bert-lstm/v7.h5')
    return model


#function to return max probablity and domain of text
def check_hier_text(model, text):
        
    b = preproc(text)
    c = get_split(b)
    d = tokenizer(c, padding=True, truncation=True, max_length=max_len)
    xt = []
    if len(b.split()) < 200:
        
        ids = d['input_ids']
        mask = d['attention_mask'] 
        toks = []
        for i in ids[0]:
            toks.append(i)
        masks = []
        for i in mask[0]:
            masks.append(i)

        if len(toks) < 200:
            for i in range(200 - len(toks)):
                toks.append(0)
                masks.append(0)
        x_t = [np.asarray(toks).astype(np.float32), np.asarray(masks).astype(np.float32)]
        
    else:
        x_t = [np.asarray(d['input_ids']), np.asarray(d['attention_mask'])]

    pred = model.predict(x_t)
    labels = pred.argmax(axis = 1)
    best = np.argmax(np.bincount(labels))
    dom = id_to_label[best]
    pr = 0
    c = 0
    for i in pred:
        if i.argmax() == best:
            pr += i.max()
            c += 1
    return pr / c, dom


