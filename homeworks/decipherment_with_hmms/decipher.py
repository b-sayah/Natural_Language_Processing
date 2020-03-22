# References:

from nltk.tag.hmm import HiddenMarkovModelTrainer, HiddenMarkovModelTagger
from nltk.probability import (LaplaceProbDist,
                              MLEProbDist,
                              ConditionalFreqDist,
                              ConditionalProbDist,
                              ConditionalProbDistI
                              )
from utils import GetCorpora, tagger_best_path, get_bigram

import numpy as np
import argparse
import os
import sys
import nltk
import string

# data_dir
base_path = 'a2-q3/a2data'

# Create the parser annd add arguments
parser = argparse.ArgumentParser()
parser.add_argument('cipher', metavar='cipher', type=str, default='cipher1',
                       help='cipher1, cipher2 or cipher3')
parser.add_argument('-laplace', action='store_true',
                       help='use laplace smoothing')

parser.add_argument('-lm', action='store_true',
                       help='improve plain text and compute bigram chars transitions')
args = vars(parser.parse_args())

cipher_folder = args['cipher']
cipher_path = os.path.join(base_path, cipher_folder)

if not os.path.isdir(cipher_path):
    print('path does not exist, the current data dir is: \n', base_path)
    print('pass one of cipher_folders: cipher1, cipher2 or cipher3')
    sys.exit()

# Loading corpora
Cipher = GetCorpora(cipher_folder, cipher_path)
train_plain, train_cipher, test_plain, test_cipher = Cipher.load_corpus()
print('{} corpora loaded'.format(cipher_folder))

# symbols and states 29 character each
Symbols = list(string.ascii_lowercase) + [',', '.', ' ']
States = Symbols

# remove line break
train_plain = [char.strip('\n') for char in train_plain]
train_cipher = [char.strip('\n') for char in train_cipher]
test_plain = [char.strip('\n') for char in test_plain]
test_cipher = [char.strip('\n') for char in test_cipher]

# return tuples as expected by HMM tagger
train_tagged_corpus = []
for s, st in zip(train_cipher, train_plain):
    sample = list(zip(s, st))
    train_tagged_corpus.append(sample)

test_tagged_corpus = []
for s, st in zip(test_cipher, test_plain):
    sample = list(zip(s, st))
    test_tagged_corpus.append(sample)

if args['laplace'] == True:
    Estimator = LaplaceProbDist
    print_estimator = 'Laplace'  # just for printing
else:
    Estimator = MLEProbDist
    print_estimator = 'MLE'  # just for printing
#/////////////// Train test MLE and la place etimsator /////////////////

# training
HMM_tagger = HiddenMarkovModelTrainer(states=States, symbols=Symbols)
HMM_tagger = HMM_tagger.train_supervised(train_tagged_corpus, estimator=Estimator)
print(HMM_tagger)

#/////////////////////// TEXT IMPROVEMENT  /////////////////////////////

if args['lm'] == True:
    # get additional text
    # Text number 2554 English translation of Crime and Punishment
    bigrams = get_bigram(train_plain, url='http://www.gutenberg.org/files/2554/2554-0.txt')
    # conditional freq dist
    cfd = ConditionalFreqDist(bigrams)
    # Conditional probability distribution
    cpd = nltk.ConditionalProbDist(cfd, Estimator)

    Trainer = nltk.tag.hmm.HiddenMarkovModelTagger(states=States,
                                                   symbols=Symbols,
                                                   priors=None,
                                                   transitions=cpd,
                                                   outputs=None
                                                   )
    HMM_tagger = Trainer.train(labeled_sequence=train_tagged_corpus, estimator=Estimator, )

    print('HMM trained on {} with text improvement and {} estimator \n'.format(cipher_folder, print_estimator))

# testing
HMM_tagger.test(test_tagged_corpus, verbose=False)
# Accuracy using best path simple
accuracy = tagger_best_path(HMM_tagger, test_cipher, test_plain)
print('Accuracy with best paths: %.2f'% accuracy)


