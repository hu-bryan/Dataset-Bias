from collections import Counter
import nltk
from nltk.tokenize import word_tokenize

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



