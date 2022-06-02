#!/usr/bin/env python
# coding: utf-8

import transformers
from transformers import BertTokenizerFast
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split as tts
import pandas as pd
import random
import numpy as np
import re
import string
from nltk.corpus import stopwords
from tensorflow.keras import models, layers, preprocessing as kprocessing
from tensorflow.keras import backend as K
from bs4 import BeautifulSoup
from bs4.element import Comment
import requests
import os
import PyPDF2
import pickle as pk

def init():

    stop_words = stopwords.words('english')
    max_len = 200   # maxium length of sentence for heirarchical transformer

    id_to_label = {0 : 'Computer', 1: 'Recreation', 2 : 'Science', 3: 'Politics', 4: 'Rfp', 5: 'Water'} # mapping from id to domain

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
    y_out = layers.Dense(len(id_to_label.keys()), activation='softmax')(out_dense)
    
    model = models.Model([idx, masks], y_out)
    
    for layer in model.layers[:3]:
        layer.trainable = False
        
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Loading pretrained weights into the model (By default v7 is being loaded, but user can load any of the available weights in the model_weights folder)
def config_model():

    model = create_model()
    model.load_weights(r'./model_weights/bert-lstm/v7.h5')
    return model

# Training the current KITE domain classification model (Multinomial NB with TFIDF counts as features) for comparison
def mnb():
    
    X_train, X_test, Y_train, Y_test = get_data()

    count_vect = CountVectorizer(stop_words = 'english')
    X_train_counts = count_vect.fit_transform(X_train)
    tfidf_transformer = TfidfTransformer()
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    clf = MultinomialNB().fit(X_train_tfidf, Y_train)
    return (count_vect, tfidf_transformer, clf)

count_vect, tfidf_transformer, clf = mnb()

def classify(text):
    
    docs_new = [text]
    X_new_counts = count_vect.transform(docs_new)
    X_new_tfidf = tfidf_transformer.transform(X_new_counts)
    predicted = clf.predict(X_new_tfidf)
    prob = clf.predict_proba(X_new_tfidf)
    # print(prob)
#     if np.max(prob) < 0.4:   #other
#         predicted[0] = 5
    if predicted[0] == 0:
        return prob, 'Computer'
    if predicted[0] == 1:
        return prob, 'Recreation'
    if predicted[0] == 2:
        return prob, 'Science'
    if predicted[0] == 3:
        return prob, 'Politics'
    if predicted[0] == 4:
        return prob, 'Rfp'
    if predicted[0] == 5:
        return prob, 'Water'

# TESTING FOR URLS
def url_init():

    sci = ['https://vikaspedia.in/education/childrens-corner/science-section/articles-on-science',
       'https://www.livescience.com/51720-photosynthesis.html',
       'https://www.urmc.rochester.edu/encyclopedia/content.aspx?ContentTypeID=160&ContentID=37#:~:text=Plasma%20is%20the%20largest%20part,carries%20water%2C%20salts%20and%20enzymes.',
       'http://hyperphysics.phy-astr.gsu.edu/hbase/pbuoy.html',
       'https://opentextbc.ca/introductorychemistry/chapter/neutralization-reactions/#:~:text=Neutralization%20is%20the%20reaction%20of%2C%20solid%20salts%2C%20and%20water.',
       'https://byjus.com/physics/thermodynamics/',      
       'https://www1.grc.nasa.gov/beginners-guide-to-aeronautics/newtons-laws-of-motion/',
       'https://byjus.com/biology/animal-husbandry-food-animals/#:~:text=Animal%20husbandry%20refers%20to%20livestock,animal%20husbandry%20for%20their%20livelihood.',
       'https://en.wikipedia.org/wiki/Physical_optics#:~:text=In%20physics%2C%20physical%20optics%2C%20or,geometric%20optics%20is%20not%20valid.',
       'https://education.nationalgeographic.org/resource/cloning'
      ]

    recr = ['https://www.ukrinform.net/rubric-sports/3428180-ukraine-ranks-second-at-beijing-paralympics-medal-tally.html',
            'https://www.rolandgarros.com/en-us/article/rg2022-day-2-live-osaka-andreescu-swiatek-nadal-djokovic',
            'https://www.ukrinform.net/rubric-sports/3418781-ipc-declines-athlete-entries-from-russia-and-belarus-for-beijing-2022.html',
            'https://www.hindustantimes.com/sports/football/uk-government-approves-sale-of-chelsea-by-sanctioned-abramovich-101653469935208.html',
            'https://www.skysports.com/tennis/live-blog/12110/12620961/emma-raducanu-in-french-open-second-round-action-at-roland-garros-live',
            'https://www.espn.in/football/soccer-transfers/story/4673400/kylian-mbappe-on-rejecting-real-madrid-my-decision-based-on-psgs-sporting-project-not-money',
            'https://indianexpress.com/article/sports/football/mohamed-salah-on-revenge-mission-in-champions-league-final-7935737/',
            'https://www.upi.com/Sports_News/MLB/2022/05/24/Cincinnati-Reds-fan-accidentally-catches-ball-in-beer-cup-chugs-for-crowd-basball-MLB/3281653399485/',
            'https://www.nytimes.com/2022/05/23/sports/baseball/roger-angell.html',
            'https://en.wikipedia.org/wiki/Amitabh_Bachchan',
        ]

    pol = ['https://www.ndtv.com/india-news/will-accept-hanging-if-indian-intel-proves-jammu-and-kashmir-separatist-yasin-malik-3007740',
        'https://www.opindia.com/2022/05/rahul-gandhi-cambridge-india-union-states-european-union/',
        'https://www.ukrinform.net/rubric-polytics/3491764-zelensky-end-of-war-depends-on-wests-position-and-russias-desire.html',
        'http://lite.cnn.com/en/article/h_f68f79dede493cb0a82201f6ea0ce293',
        'https://www.usatoday.com/story/news/education/2022/05/20/michigan-teacher-barack-obama-primates/9860121002/',
        'https://en.wikipedia.org/wiki/Arnold_Schwarzenegger',
        'https://en.wikipedia.org/wiki/Joe_Biden',
        'https://timesofindia.indiatimes.com/india/8-years-8-political-highlights-how-bjp-has-fared-in-modi-era/articleshow/91892637.cms',
        'https://thewire.in/politics/jk-national-panthers-party-founder-bhim-singh-passes-away',
        'https://www.usatoday.com/story/news/nation/2022/05/28/uvalde-texas-school-shooter/9960611002/?gnt-cfr=1',
        ]

    comp = ['https://scholar.google.com/citations?user=mG4imMEAAAAJ&hl=en',
            'https://www.geeksforgeeks.org/compiler-construction-tools/',
            'https://www.tutorialspoint.com/operating_system/os_overview.htm',
            'https://www.w3schools.com/charsets/ref_html_ascii.asp#:~:text=The%20ASCII%20Character%20Set&text=ASCII%20is%20a%207%2Dbit,are%20all%20based%20on%20ASCII.',
            'https://www.ibm.com/cloud/learn/machine-learning#:~:text=Machine%20learning%20is%20a%20branch,rich%20history%20with%20machine%20learning.',
            'https://www.geeksforgeeks.org/machine-learning/',
            'https://www.financialexpress.com/industry/technology/apple-realityos-arvr-headset-trademark/2544036/',
            'https://www.iwmbuzz.com/tech/forgotten-your-email-password-no-worries-heres-how-to-reset-your-email-password/2022/05/31',
            'https://www.microsoft.com/security/blog/2022/05/27/android-apps-with-millions-of-downloads-exposed-to-high-severity-vulnerabilities/',
            'https://hackaday.com/2022/05/28/linux-and-c-in-the-browser/',
        ]

    water = ['https://www.epa.gov/dwreginfo/public-water-system-supervision-program-water-supply-guidance-manual',
            'https://www.cdc.gov/healthywater/drinking/public/regulations.html',
            'https://www.epa.gov/sdwa/secondary-drinking-water-standards-guidance-nuisance-chemicals',
            'https://iwa-network.org/projects/water-policy-and-regulation/',
            'https://en.wikipedia.org/wiki/Water_pollution',
            'https://www.consumernotice.org/environmental/water-contamination/',
            'https://www.hsph.harvard.edu/ehep/82-2/',
            'https://www.thermofisher.com/in/en/home/industrial/environmental/environmental-learning-center/water-quality-analysis-information/water-regulations.html',
            'https://www.waterregsuk.co.uk/guidance/legislation/what-are-water-regulations/',
            'https://www.indiawaterportal.org/articles/indian-standard-drinking-water-bis-specifications-10500-1991',
            ]

    urls = {'Science' : sci, 'Recreation' : recr, 'Politics' : pol, 'Computer' : comp, 'Water' : water}
    return urls

# Helper function for get_data_url
def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'footer', 'header']:
        return False
    if isinstance(element, Comment):
        return False
    return True

# Scraping and preprocessing text data from given URL
def get_data_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
    x = requests.get(url, headers = headers)
    if x.status_code != 200:
        return "Access Denied, Status Code : {}".format(x.status_code)
    soup = BeautifulSoup(x.content, 'html.parser')
    text = soup.findAll(text=True)
    visible_texts = filter(tag_visible, text)  
    return u" ".join(t.strip() for t in visible_texts)

# Function to return prediction for Transformer Model
# In case of multiple splits of data, each split of 200 length is predicted independantly and maximum occuring label is outputted
def test_hier_url(model, t):
    
    b = preproc(t)
    c = get_split(b)
    d = tokenizer(c, padding=True, truncation=True, max_length=max_len)
    x_t = [np.asarray(d['input_ids']), np.asarray(d['attention_mask'])]
    pred = model.predict(x_t)
    labels = pred.argmax(axis = 1)
    return id_to_label[np.argmax(np.bincount(labels))] 
 
# Testing old KITE model on unseen URL
def test_mnb_url(a):

    b = classify(a)
    return b[1], np.max(b[0])

# Compiling unseen URL results for both models
def check_unseen(model, urls, v):
    
    fname = r'./results/url_data_' + v + '.csv'
    f = open(fname, 'w')
    f.write('URL, DOMAIN, OLD,' + v + '\n')
    
    for dom in urls.keys():
        
        correct_nn = 0
        correct_mn = 0
        total = 0

        for url in urls[dom]:     
            
            temp = url
            if url.find(',') != -1:
                temp = 'NA'
                
            s = ''
            s += temp + ',' + dom + ','
            a = get_data_url(url)
            
            if a.startswith('Access Denied'):
                f.write(a)
                continue
            
            total += 1
            mn_op = test_mnb_url(a)
            s += mn_op[0] + ','
            nn_op = test_hier_url(model, a) 
            s += nn_op 
            if mn_op[0] == dom:
                correct_mn += 1
            if nn_op == dom:
                correct_nn += 1
            f.write(s + '\n')
        
        print('Checking URLs for ' + dom)
        print('old kite : ' + str(correct_mn) + '/' + str(total))
        print(v + ' : ' + str(correct_nn) + '/' + str(total))
        
    print('\n..... '  + fname + ' created......\n')
    f.close()
    return fname

# Individually testing one URL on Hierarchical Transformer model
def check_hier_url(model, url):
    
    a = get_data_url(url)
    if a.startswith('Access Denied'):
        return "Access Denied"
    b = preproc(a)
    c = get_split(b)
    d = tokenizer(c, padding=True, truncation=True, max_length=max_len)
    x_t = [np.asarray(d['input_ids']), np.asarray(d['attention_mask'])]
    pred = model.predict(x_t)
    labels = pred.argmax(axis = 1)
    return id_to_label[np.argmax(np.bincount(labels))] 
