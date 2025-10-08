from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from gensim.models import Word2Vec
import numpy as np


def embedding(corpus):
    sentences = []
    for uttr in corpus:
        tokens = word_tokenize(uttr.lower())
        sentence = []
        for tok in tokens:
            if tok.isalpha():
                sentence.append(tok)
        sentences.append(sentence)
    return Word2Vec(sentences=sentences, vector_size=100, window=10, sg=1)

def classifier_pmi(corpus, id_terms, labels, num_classes):
    N = len(corpus)
    id_count = 0
    labels_pair_count = np.zeros(num_classes)
    labels_single_count = np.zeros(num_classes)

    for i, uttr in enumerate(corpus):
        label = labels[i]  #np.array 
        labels_single_count += label
        for id_term in id_terms:
            if id_term in word_tokenize(uttr.lower()):
                id_count += 1
                labels_pair_count += label
                break 
    
    pmis = (np.log2(N * labels_pair_count) 
            - np.log2(id_count * labels_single_count)
            ).tolist()
    return pmis

        


def process(corpus, stopwords):
    """
    args: 
        corpus - a dictionary of strings
        stopwords - a set of tokens, where each token is a stop word

    returns: 
        processed_corpus - a list of sets, where each set consists of unique 
        lowercase alphabetical tokens and excludes stop words and punctuations
    """

    processed_corpus = dict()

    for idx, uttr in corpus.items():
        tokens = word_tokenize(uttr.lower())
        uttr_words = set()
        for tok in tokens:
            if tok.isalpha() and tok not in stopwords:
                uttr_words.add(tok)
        processed_corpus[idx] = uttr_words
        
    return processed_corpus


def count_words(corpus):
    """
    args: 
        corpus - an dictionary of sets, where each set consists of unique word 
        tokens

    returns:
        count - a Counter object such that count[w] is the number of utterances
        that contains word w
    """
    count = Counter()
    for uttr in corpus.values():
        for word in uttr:
            count[word] += 1
    return count



