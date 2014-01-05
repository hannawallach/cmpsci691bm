import re
from csv import reader
from numpy import argsort, array, cumsum, log, ones, prod, searchsorted, zeros
from numpy.random import uniform
from numpy.random.mtrand import dirichlet
from scipy.special import gamma, gammaln

from vocabulary import *


def sample(dist, num_samples=1):
    """
    Uses the inverse CDF method to return samples drawn from an
    (unnormalized) discrete distribution.

    Arguments:

    dist -- (unnormalized) distribution

    Keyword arguments:

    num_samples -- number of samples to draw
    """

    cdf = cumsum(dist)
    r = uniform(size=num_samples) * cdf[-1]

    return cdf.searchsorted(r)


def generate_corpus(beta, mean, N):
    """
    Returns a corpus of tokens drawn from a Dirichlet--multinomial
    unigram language model. Each token is an instance of one of V
    unique word types, represented by indices 0, ..., V - 1.

    Arguments:

    beta -- concentration parameter for the Dirichlet prior
    mean -- V-dimensional mean of the Dirichlet prior
    N -- number of tokens to generate
    """

    pass # YOUR CODE GOES HERE


def generate_corpus_collapsed(beta, mean, N):
    """
    Returns a corpus of tokens drawn from a Dirichlet--multinomial
    unigram language model using the 'collapsed' generative process
    (i.e., phi is not explicitly represented). Each token is an
    instance of one of V unique word types.

    Arguments:

    beta -- concentration parameter for the Dirichlet prior
    mean -- V-dimensional mean of the Dirichlet prior
    N -- number of tokens to generate
    """

    V = len(mean) # vocabulary size

    corpus = zeros(N, dtype=int) # corpus

    Nv = zeros(V, dtype=int) # counts for each word type

    pass # YOUR CODE GOES HERE


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


def preprocess(filename, stopword_filename=None):
    """
    Preprocesses a CSV file and returns a tuple consisting of a
    bijective mapping from unique word types to indices and a list of
    tokens, represented by indices 0, ..., V - 1.

    Arguments:

    filename -- name of CSV file
    stopword_filename -- name of file containing stopwords
    """

    text = []

    for _, _, description in reader(open(filename), delimiter='\t'):
        text.append(description)

    text = ' '.join(text)

    stopwords = create_stopword_list(stopword_filename)

    pass # YOUR CODE GOES HERE


def evidence_1(corpus, beta, mean):
    """
    Returns the evidence for a corpus of tokens according to a
    Dirichlet--multinomial unigram language model.

    Arguments:

    corpus -- list of tokens, represented by indices 0, ..., V - 1
    beta -- concentration parameter for the Dirichlet prior
    mean -- V-dimensional mean of the Dirichlet prior
    """

    N = len(corpus) # number of tokens in corpus
    V = len(mean) # vocabulary size

    Nv = zeros(V, dtype=int)

    for n in xrange(N):
        Nv[corpus[n]] += 1

    pass # YOUR CODE GOES HERE


def evidence_2(corpus, beta, mean):

    N = len(corpus)
    V = len(mean)

    Nv = zeros(V, dtype=int)

    pass # YOUR CODE GOES HERE


def log_evidence_1(corpus, beta, mean):
    """
    Returns the log evidence for a corpus of tokens according to a
    Dirichlet--multinomial unigram language model.

    Arguments:

    corpus -- list of tokens, represented by indices 0, ..., V - 1
    beta -- concentration parameter for the Dirichlet prior
    mean -- V-dimensional mean of the Dirichlet prior
    """

    N = len(corpus)
    V = len(mean)

    Nv = zeros(V, dtype=int)

    for n in xrange(N):
        Nv[corpus[n]] += 1

    pass # YOUR CODE GOES HERE


def log_evidence_2(corpus, beta, mean):

    N = len(corpus)
    V = len(mean)

    Nv = zeros(V, dtype=int)

    pass # YOUR CODE GOES HERE


def posterior_mean(corpus, beta, mean):
    """
    Returns the mean of the posterior distribution over phi.

    Arguments:

    corpus -- list of tokens, represented by indices 0, ..., V - 1
    beta -- concentration parameter for the Dirichlet prior
    mean -- V-dimensional mean of the Dirichlet prior
    """

    N = len(corpus)
    V = len(mean)

    Nv = zeros(V, dtype=int)

    for n in xrange(N):
        Nv[corpus[n]] += 1

    pass # YOUR CODE GOES HERE


def print_top_types(vocab, corpus, beta, mean, num=10):
    """
    Prints the most probable word types according to the mean of the
    posterior distribution over phi.

    Arguments:

    vocab -- bijective mapping from unique word types to indices
    corpus -- list of tokens, represented by indices 0, ..., V - 1
    beta -- concentration parameter for the Dirichlet prior
    mean -- V-dimensional mean of the Dirichlet prior

    Keyword arguments:

    num -- number of types to print
    """

    top_types = map(vocab.lookup, argsort(posterior_mean(corpus, beta, mean)))
    print ' '.join(top_types[-num:][::-1])


def log_predictive_prob(new_corpus, corpus, beta, mean):
    """
    Returns the log predictive probability of a new corpus of tokens
    given an existing corpus of tokens according to a
    Dirichlet--multinomial unigram langauge model.

    Arguments:

    new_corpus -- list of tokens, represented by indices 0, ..., V - 1
    corpus -- list of tokens, represented by indices 0, ..., V - 1
    beta -- concentration parameter for the Dirichlet prior
    mean -- V-dimensional mean of the Dirichlet prior
    """

    pass # YOUR CODE GOES HERE
