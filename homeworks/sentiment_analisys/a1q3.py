
"""
References:

https://www.nltk.org/book/ch06.html

https://www.datacamp.com/community/tutorials/text-analytics-beginners-nltk

https://www.machinelearningplus.com/nlp/lemmatization-examples-python/

please note that you may change data_dir to access data files and execute the code
line 98

developped on colab
"""


import nltk
import numpy as np
import pandas as pd
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet, stopwords
from nltk.tokenize import RegexpTokenizer, sent_tokenize, word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.dummy import DummyClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import seaborn as sns


def get_data(path):
    
    with open(os.path.join(path, "rt-polarity.pos"), encoding='latin-1', mode='r') as f:
        pos_review = f.readlines()
    with open(os.path.join(path, "rt-polarity.neg"), encoding='latin-1', mode='r') as f:
        neg_review = f.readlines()

    snipets = np.concatenate([np.array(pos_review), np.array(neg_review)])
    categories = np.zeros(len(snipets))
    categories[:len(pos_review)] = 1

    return snipets, categories


# Stemming
def stemming(snippet):
    stem = PorterStemmer()
    stemmed_review = []
    tokens = word_tokenize(snippet)
    for token in tokens:
        stemmed_token = stem.stem(token)
        stemmed_review.append(stemmed_token)
    return ' '.join(stemmed_review)


# Lemmatize with POS Tag
def get_pos_tag(word):
    # '[0][1][0].upper()' return compatible word-net or hashable type
    pos_tag = nltk.pos_tag([word])[0][1][0].upper()
    # mappping
    tag_dict = {"N": wordnet.NOUN, "V": wordnet.VERB,
                'J': wordnet.ADJ, "R": wordnet.ADV}
    return tag_dict.get(pos_tag, wordnet.VERB) 


def lemmatization(snippet):
    wn_lemma = WordNetLemmatizer()
    tokens = word_tokenize(snippet)
    lemma_review = []
    punctuations = '''!()-[]{};:'",<>./?@#$%^&*_~[]'''
    for token in tokens:
        if token not in punctuations:
            lemma_token = wn_lemma.lemmatize(token, get_pos_tag(token))
            lemma_review.append(lemma_token)
    return ' '.join(lemma_review)


def fit_and_predict(classifier):
    classifier.fit(x_train, y_train)
    predicts = classifier.predict(x_test)
    acc = metrics.accuracy_score(y_test, predicts)
    print('{} accuracy: {}'.format(type(classifier).__name__, acc))
    conf_matrix = confusion_matrix(y_test, predicts)
    return acc, conf_matrix


# MAIN
if __name__ == '__main__':
    # Data loading, data_dir is cwd considering files in the same folde
    data_dir = ""
    print('data_dir', data_dir)
    lines, labels = get_data(data_dir)

    print('downloading necessary ressource for stemming, lemmatization and english stopwords list...')
    # necessary ressource for stemming
    nltk.download('punkt')
    # ressource for lemmatization
    nltk.download('averaged_perceptron_tagger')
    nltk.download('wordnet')
    # nltk stopwords list for the feature extraction
    nltk.download('stopwords')

    # Stemming all sentences in stemmed_reviews
    reviews = lines
    stemmed_reviews = []
    print("stemming...")
    for review in reviews:
        stemmed_reviews.append(stemming(review))

    # lemmatization of all sentences in lemmatized_reviews
    lemmatized_reviews = []
    print("lemmatization...")
    for review in reviews:
        lemmatized_reviews.append(lemmatization(review))

    # for reproductibiilty
    seed = 1234

    # Dictionary of classifiers and their parameters tuned separatly
    classifier_sets = {"logis_reg": LogisticRegression(penalty='l2', C=0.92,
                                                       solver='saga',
                                                       random_state=seed,
                                                       max_iter=250),

                       "svm": LinearSVC(random_state=seed, C=0.035),

                       "naive_bayes": MultinomialNB(alpha=1.),
                       "baseline": DummyClassifier(strategy='uniform',
                                                   random_state=seed)

                       }

    # tokenizer to remove non desired characters
    pattern = r'[a-zA-Z0-9]+'
    Token = RegexpTokenizer(pattern)
    # threshold ranges to remove frequent and infrequent words
    Min_df = [0.0002, 0.0003, 0.0004, 0.001, 0.002, 0.003, 1]
    Max_df = [0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4]
    best_results = []
    preprocess_reviews = [stemmed_reviews, lemmatized_reviews]

    print('classification...')
    for clf_name, clf in classifier_sets.items():
        print("============================={}==========================\n".format(clf_name))
        for reviews in preprocess_reviews:
            best_acc = 0
            for mn in Min_df:
                for mx in Max_df:
                    # feature extraction
                    cv = CountVectorizer(ngram_range=(1, 1),
                                         tokenizer=Token.tokenize,
                                         min_df=mn,
                                         max_df=mx
                                         # stop_words='english',
                                         # stop_words=stopwords.words('english')
                                         )
                    x_feat = cv.fit_transform(reviews)

                    # Split data
                    x_train, x_test, y_train, y_test = train_test_split(x_feat, labels,
                                                                        test_size=.2,
                                                                        random_state=23)
                    #  classifiers training and prediction
                    accuracy, conf_matrix = fit_and_predict(clf)

                    if best_acc < accuracy:
                        best_acc = accuracy
                        print("confusion matrix\n", conf_matrix)
                        # for printing only
                        if reviews == lemmatized_reviews:
                            preproces_type = "lemmatization"
                        else:
                            preproces_type = "stemming"
                        # store best results and relevant parameters
                        best_result = ({'Classifier': type(clf).__name__,
                                        'Preprocessing': preproces_type,
                                        'Best_accuracy': best_acc,
                                        'min_df': mn, 'max_df': mx,
                                        'confusion_matrix': conf_matrix,
                                        'model parameters': clf.get_params})

                    print("min_df= {} , max_df= {}'\n".format(mn, mx))

            print('the best model and its parameters\n', best_result)
            best_results.append(best_result)

    df = pd.DataFrame(best_results)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df[['Classifier', 'Preprocessing', 'Best_accuracy',
                  'min_df', 'max_df', 'confusion_matrix']])
    df.to_csv(index=False)
    sns.barplot(x='Classifier', y='Best_accuracy', hue='Preprocessing', data=df)

