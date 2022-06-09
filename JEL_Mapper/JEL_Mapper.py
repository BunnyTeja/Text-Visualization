#!/usr/bin/env python
# coding: utf-8

import requests
from bs4 import BeautifulSoup
from bs4.element import Comment
import pickle as pk
from fuzzywuzzy import fuzz
from nltk import bigrams, ngrams
from nltk.tokenize import word_tokenize


def tag_visible(element):
    if element.parent.name in ['style', 'script', 'head', 'title', 'meta', '[document]', 'footer', 'header']:
        return False
    if isinstance(element, Comment):
        return False
    return True

def get_data_url(url):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
    x = requests.get(url, headers = headers)
    if x.status_code != 200:
        return "Access Denied, Status Code : {}".format(x.status_code)
    soup = BeautifulSoup(x.content, 'html.parser')
    text = soup.findAll(text=True)
    visible_texts = filter(tag_visible, text)  
    #return visible_texts
    return u" ".join(t.strip() for t in visible_texts)


def load_data():

    with open(r'./data/jel_code_to_term.pk', 'rb') as f:
        code_to_term = pk.load(f)
    with open(r'./data/jel_term_to_code.pk', 'rb') as g:
        term_to_code = pk.load(g)

    return code_to_term, term_to_code

# Generate a trie from a map from words to codes
def get_trie(m1):
    trie = {}
    for word in m1.keys():
        if word == '':
            continue
        temp = trie
        for letter in word:
            if letter not in temp.keys():
                temp[letter] = {}
            temp = temp[letter]
        temp['__end__'] = m1[word]
    return trie

# Return the closest string to the word in the trie
def find_(trie, word):
    c = ''
    temp = trie
    for letter in word:
        if letter not in temp.keys():
            dist, word = get_nearest(temp)
            return (dist, c+word)
        else:
            c += letter
            temp = temp[letter]
    if '__end__' not in temp.keys():
        dist, word = get_nearest(temp)
        return (dist, c+word)
    else:
        return (0, c)

# Helper function for find_
def get_nearest(trie):
    ans = 1000000
    word = ''
    if '__end__' in trie.keys():
        return (0, '')
    for key in list(trie.keys()):
        temp = get_nearest(trie[key])
        if ans > temp[0]:
            ans = temp[0]
            word = key+temp[1]
    return (ans+1, word)

# For eg, breaking 'B10' to 'B', 'B1' and 'B10'
def break_codes(code):
    size = len(code)
    hier = []
    while code[:size] != '' :
        hier.append(code[:size])
        size -= 1
    return hier

# Reutrn Classification Codes, given a piece of text
def get_codes_class(text, term_to_code, code_to_term, triee, to_print = 0):
    
    try:
        # A map from a term to its count
        ans = {}

        # Tokenize text into words
        tokens = [str(word).lower() for word in word_tokenize(text) if str(word).isalpha()]
        
        # A map to check whether a word has already been included in the Classification
        done = {}

        # Generate N-Grams from 7 to 2
        for i in range(7,1,-1):
            text_temp_ = list(ngrams(tokens, i))
            for j, el in enumerate(text_temp_):
                term = ' '.join(el)
                dist, nearest = find_(triee, term)
                similarity = fuzz.ratio(nearest.strip(), term.strip())  #min edit distance

                # Consider terms if similarity > 85 in the trie
                if similarity >= 85:
                    temp = term
                    term = nearest
                    if term != 'general' and term != '' and term != 'other':
                        check = False
                        for k in range(len(el)):   #len(el) : n of ngram
                            if done.get(j+k, -1) == -1:
                                check = True
                                break
                        if check:
                            for k in range(len(el)):
                                done[j+k] = 1
                            if to_print == 1:
                                print(temp, ',', nearest, ',', similarity, ',', j, ',', term_to_code[term])
                            ans[term] = ans.get(term, 0)+1
        
        # Rest of the code decides what to do in case a term belongs to 2 codes

        final = {}  #final code for a term if multiple codes present  (term, code) : count
        codes = {}  #count of a code

        for key in ans.keys():
            
            if len(term_to_code[key]) == 1:  # if only one code for term
                
                final[(key, list(term_to_code[key])[0])] = ans[key]
                code = list(term_to_code[key])[0]
                hier_codes = break_codes(code)
                for i in hier_codes:
                    codes[i] = codes.get(i, 0) + 1
              
            else: # more than one code for the term
                
                lengths = []
                for el in term_to_code[key]:   # iterating over all possible codes for the term
                    lengths.append(codes.get(el, 0))
                maxlen = max(lengths)
                max_codes = []
                for el in term_to_code[key]:
                    if codes.get(el, 0) == maxlen:
                        max_codes.append(el)
                
                maxlen2 = 0
                best_code = ''
                for c in max_codes:
                    length = codes.get(c[:-1], 0) # eg: B12 and C01 have equal count, checking counts for B1 and C0
                    if length > maxlen2:
                        maxlen2 = length
                        best_code = c
                
                hier_codes = break_codes(best_code)
                for i in hier_codes:
                    codes[i] = codes.get(i, 0) + 1
                
                final[(key, c)] = ans[key]
                
        # finding top10 codes in the text and the code classification (letter) with the most the most codes in the text       
        codes = {}
        title = {}
        for key in list(final.keys()):
            title[key[1][0]] = title.get(key[1][0], 0)+1
        letter = max(title, key= lambda x: title[x])
        title = code_to_term[letter]
        final_codes = sorted(list(final.keys()), key = lambda x: final[x], reverse=True)[:10]
        areas = {}
        codes = {}
        for i, code in enumerate(final_codes):
            codes[code[1]] = final[code]
            areas[code_to_term[code[1]]] = codes[code[1]]

        # Letter is the Code for the Letter with most codes (A-K)
        # codes is a map from Classification Code to Frequency (Top 10 codes)
        return (letter, codes)
    except:
        return ("", {})

