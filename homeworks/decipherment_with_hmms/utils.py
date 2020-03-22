# reference:  https://www.nltk.org/book/ch03.html

import glob
import os
import itertools
import nltk
import string
from nltk.tokenize import sent_tokenize
from nltk.tokenize.punkt import PunktSentenceTokenizer, PunktParameters
from urllib import request


class GetCorpora:

    def __init__(self, cipher, cipher_path):
        self.cipher = cipher
        self.cipher_path = cipher_path

    def load_corpus(self):
        files = sorted(glob.glob(os.path.join(self.cipher_path, '*')), reverse=True)
        corpora = []
        for file in files:
            with open(file, mode='r') as f:
                corpus = f.readlines()
            corpora.append(corpus)
        return corpora

def remove_chars(sentence):
    '''
     keep necessary chars
    :param sentence: receive a sentence segmented from raw text
    :return: cleaned sentences with the 29 symbols
    '''
    all_chars = string.ascii_lowercase + ' ' + ',' + '.'
    return ''.join(char for char in sentence if char in all_chars )


def get_bigram(plain_text, url):
    '''
    take an url and a plain text. Preprocessing
    added text, segmentation, lower and keep
    the 29 characters. get conditional frequence
    distribution of bigrams of the new plain text
    :param plain_text: given plain text
    :param url: url
    :return: ConditionalFreqDist of bigrams (combined plain text
    '''
    print('downloading and processing text from ', url)
    added_text = request.urlopen(url)
    added_text = added_text.read().decode('utf8')
    # segmentation
    sent_detector = nltk.data.load('tokenizers/punkt/english.pickle')
    sentences = sent_detector.tokenize(added_text.strip('\n'))

    # remove \n \r and lower text
    process_sentence = [sent.replace('\r', '').replace('\n', '').lower() for sent in sentences]

    # remove non desired characters
    cleaned_text = [remove_chars(line) for line in process_sentence]
    cleaned_text = cleaned_text + plain_text

    # get bigram
    bigrams = []
    for sent in cleaned_text:
        bigrams += list(nltk.bigrams(sent))
    return bigrams


def tagger_best_path(tagger, states, symbols):
    '''get accuracy using best path simple (Viterbi)'''
    correct = 0
    total = 0

    for ob, hid in zip(states, symbols):
        predicted= tagger.best_path_simple(ob)
        for pred, h in zip(predicted, hid):
            if (pred == h):
                correct += 1
            total += 1

    acc = correct/total * 100
    return acc



