import re
from csv import reader

from corpus import Corpus


def create_stopword_list(f):
    """
    Returns a set of stopwords.

    Arguments:

    f -- list of stopwords or name of file containing stopwords
    """

    if not f:
        return set()

    if isinstance(f, basestring):
        f = file(f)

    return set(word.strip() for word in f)


def tokenize(text, stopwords=set()):
    """
    Returns a list of lowercase tokens corresponding to the specified
    string with stopwords (if any) removed.

    Arguments:

    text -- string to tokenize

    Keyword arguments:

    stopwords -- set of stopwords to remove
    """

    tokens = re.findall('[a-z]+', text.lower())

    return [x for x in tokens if x not in stopwords]


def preprocess(filename, stopword_filename=None, extra_stopwords=None):
    """
    Preprocesses a CSV file and returns ...

    Arguments:

    filename -- name of CSV file

    Keyword arguments:

    stopword_filename -- name of file containing stopwords
    extra_stopwords -- list of additional stopwords
    """

    stopwords = create_stopword_list(stopword_filename)
    stopwords.update(create_stopword_list(extra_stopwords))

    corpus = Corpus()

    for fields in reader(open(filename), delimiter='\t'):
        corpus.add(fields[0], tokenize(fields[-1], stopwords))

    corpus.freeze()

    return corpus
