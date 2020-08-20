import os
import re
import nltk
import numpy as np
from sklearn import feature_extraction
from tqdm import tqdm
import codecs
#from embeddings import get_similarity_vector
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.spatial import distance
_wnl = nltk.WordNetLemmatizer()
import pickle
from utils.score import LABELS
from utils.system import parse_params, check_version
from utils.dataset import DataSet


def normalize_word(w):
    return _wnl.lemmatize(w).lower()


def get_tokenized_lemmas(s):
    return [normalize_word(t) for t in nltk.word_tokenize(s)]


def clean(s):
    # Cleans a string: Lowercasing, trimming, removing non-alphanumeric

    return " ".join(re.findall(r'\w+', s, flags=re.UNICODE)).lower()


def remove_stopwords(l):
    # Removes stopwords from a list of tokens
    return [w for w in l if w not in feature_extraction.text.ENGLISH_STOP_WORDS]


def generate_baseline_feats(feat_fn, headlines, bodies, feature_file):
    if feature_file =="": # this means that this request is for the deployed model to rpedict one instance from a user, therefore, we do not need to save the extracted features
        feats = feat_fn(headlines, bodies)
        return feats
    else: # this is for training a model on a dataset
        if not os.path.isfile(feature_file): #if features are not stored as files, calculate them and store them in files then load the files
            feats = feat_fn(headlines, bodies)
            np.save(feature_file, feats)
        return np.load(feature_file)

def generate_additional_features(lexicon_file, headlines, bodies, feature_file):
    if feature_file == "" and not lexicon_file =="tfidf" :  # this means that this request is for the deployed model to rpedict one instance from a user, therefore, we do not need to save the extracted features
        feats = lexical_features(headlines, bodies,lexicon_file )
        return feats
    elif feature_file == "" and lexicon_file =="tfidf":
        feats = tfidf_features(headlines, bodies )
        return feats
    else:   # this is for training a model on a dataset
        if not os.path.isfile(feature_file): #if features are not stored as files, calculate them and store them in files then load the files
            #feats = feat_fn(headlines, bodies)
            if lexicon_file !="" and lexicon_file !="embeddings" and  lexicon_file !="tfidf":
                feats = lexical_features(headlines, bodies,lexicon_file )
                np.save(feature_file, feats)

            # if lexicon_file =="embeddings":
            #     feats = embeddings_features(headlines, bodies )
            #     np.save(feature_file, feats)

            if lexicon_file =="tfidf":
                feats = tfidf_features(headlines, bodies )
                np.save(feature_file, feats)

        return np.load(feature_file)

# def embeddings_features(h,b):
#     X = get_similarity_vector(h,b)
#     return X

def get_corpus():

    check_version()
    parse_params()

    # Load the training dataset and generate folds
    d = DataSet()
    # Load the competition dataset
    competition_dataset = DataSet("competition_test")

    # Merging the train, test to train the chosen model on the full dataset
    d.articles.update(competition_dataset.articles)
    d.stances.extend(competition_dataset.stances)

    h, b, y = [], [], []

    for stance in d.stances:
        y.append(LABELS.index(stance['Stance']))
        h.append(stance['Headline'])
        b.append(d.articles[stance['Body ID']])

    corpus = []
    corpus.extend(b)
    corpus.extend(h)

    return corpus

def tfidf_features(headlines,bodies):
    print(len(headlines))
    print(len(bodies))
    f = open("tfidf_vectorizer.pickle", "rb")
    vectorizer = pickle.load(f)
    clean_bodies = [clean(body) for body in bodies]
    clean_headlines = [clean(headline) for headline in headlines]
    X_b = vectorizer.transform(clean_bodies)
    X_h = vectorizer.transform(clean_headlines)

    similarities=[]
    shape = X_h.get_shape()
    num_of_rows = shape[0]
    #iterating over the rows of the two sparse matrices and calculating their similariy
    for i in range(0,num_of_rows):
        similarities.append(1 - (distance.cosine(X_b[i].toarray(), X_h[i].toarray())))

    return similarities


def word_overlap_features(headlines, bodies):
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        clean_headline = get_tokenized_lemmas(clean_headline)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [
            len(set(clean_headline).intersection(clean_body)) / float(len(set(clean_headline).union(clean_body)))]
        X.append(features)
    return X


def refuting_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        # 'refute',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_headline = get_tokenized_lemmas(clean_headline)
        features = [1 if word in clean_headline else 0 for word in _refuting_words]
        X.append(features)
    return X

def lexical_features(headlines, bodies,lexicon_file):
    _cue_words = []
    with codecs.open(lexicon_file,'r',encoding='utf') as f:
        lines = f.readlines()
        for line in lines:
            line= line.replace('\n','')
            cue_word = line.replace('\r','')
            _cue_words.append(cue_word)
        f.close()
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_body = clean(body)
        clean_body = get_tokenized_lemmas(clean_body)
        features = [1 if word in clean_body else 0 for word in _cue_words]
        X.append(features)
    return X

def polarity_features(headlines, bodies):
    _refuting_words = [
        'fake',
        'fraud',
        'hoax',
        'false',
        'deny', 'denies',
        'not',
        'despite',
        'nope',
        'doubt', 'doubts',
        'bogus',
        'debunk',
        'pranks',
        'retract'
    ]

    def calculate_polarity(text):
        tokens = get_tokenized_lemmas(text)
        return sum([t in _refuting_words for t in tokens]) % 2
    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        clean_headline = clean(headline)
        clean_body = clean(body)
        features = []
        features.append(calculate_polarity(clean_headline))
        features.append(calculate_polarity(clean_body))
        X.append(features)
    return np.array(X)


def ngrams(input, n):
    input = input.split(' ')
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def chargrams(input, n):
    output = []
    for i in range(len(input) - n + 1):
        output.append(input[i:i + n])
    return output


def append_chargrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in chargrams(" ".join(remove_stopwords(text_headline.split())), size)]
    grams_hits = 0
    grams_early_hits = 0
    grams_first_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
        if gram in text_body[:100]:
            grams_first_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    features.append(grams_first_hits)
    return features


def append_ngrams(features, text_headline, text_body, size):
    grams = [' '.join(x) for x in ngrams(text_headline, size)]
    grams_hits = 0
    grams_early_hits = 0
    for gram in grams:
        if gram in text_body:
            grams_hits += 1
        if gram in text_body[:255]:
            grams_early_hits += 1
    features.append(grams_hits)
    features.append(grams_early_hits)
    return features


def hand_features(headlines, bodies):

    def binary_co_occurence(headline, body):
        # Count how many times a token in the title
        # appears in the body text.
        bin_count = 0
        bin_count_early = 0
        for headline_token in clean(headline).split(" "):
            if headline_token in clean(body):
                bin_count += 1
            if headline_token in clean(body)[:255]:
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def binary_co_occurence_stops(headline, body):
        # Count how many times a token in the title
        # appears in the body text. Stopwords in the title
        # are ignored.
        bin_count = 0
        bin_count_early = 0
        for headline_token in remove_stopwords(clean(headline).split(" ")):
            if headline_token in clean(body):
                bin_count += 1
                bin_count_early += 1
        return [bin_count, bin_count_early]

    def count_grams(headline, body):
        # Count how many times an n-gram of the title
        # appears in the entire body, and intro paragraph

        clean_body = clean(body)
        clean_headline = clean(headline)
        features = []
        features = append_chargrams(features, clean_headline, clean_body, 2)
        features = append_chargrams(features, clean_headline, clean_body, 8)
        features = append_chargrams(features, clean_headline, clean_body, 4)
        features = append_chargrams(features, clean_headline, clean_body, 16)
        features = append_ngrams(features, clean_headline, clean_body, 2)
        features = append_ngrams(features, clean_headline, clean_body, 3)
        features = append_ngrams(features, clean_headline, clean_body, 4)
        features = append_ngrams(features, clean_headline, clean_body, 5)
        features = append_ngrams(features, clean_headline, clean_body, 6)
        return features

    X = []
    for i, (headline, body) in tqdm(enumerate(zip(headlines, bodies))):
        X.append(binary_co_occurence(headline, body)
                 + binary_co_occurence_stops(headline, body)
                 + count_grams(headline, body))


    return X

