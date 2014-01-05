import re, time
from csv import reader
from numpy import argsort, array, cumsum, log, ones, searchsorted, zeros
from numpy.random import uniform
from numpy.random.mtrand import dirichlet
from scipy.special import gammaln

from corpus import *
from iterview import iterview
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


def generate_corpus(alpha, m, beta, n, D, Nd):
    """
    Returns a grouped corpus drawn from a mixture of
    Dirichlet--multinomial unigram language models.

    Arguments:

    alpha -- concentration parameter for the Dirichlet prior over theta
    m -- T-dimensional mean of the Dirichlet prior over theta
    beta -- concentration parameter for the Dirichlet prior over phis
    n -- V-dimensional mean of the Dirichlet prior over phis
    D -- number of documents to generate
    Nd -- number of tokens to generate per document
    """

    corpus = GroupedCorpus()

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


def preprocess(filename, stopword_filename=None, idx=None):
    """
    Preprocesses a CSV file and returns a grouped corpus, where each
    document's group is determined by field number 'idx'. If 'idx' is
    None, all documents are assumed to belong to a single group.

    Arguments:

    filename -- name of CSV file

    Keyword arguments:

    stopword_filename -- name of file containing stopwords
    idx -- field number (e.g., 0, 1, ...) of the group
    """

    stopwords = create_stopword_list(stopword_filename)

    corpus = GroupedCorpus()

    for fields in reader(open(filename), delimiter='\t'):

        if idx:
            group = fields[idx]
        else:
            group = 'group 1'

        corpus.add(fields[0], group, tokenize(fields[-1], stopwords))

    return corpus


def log_evidence_1(corpus, alpha, m, beta, n):
    """
    Returns the log evidence for a grouped corpus (i.e., document
    tokens and groups) according to a mixture of
    Dirichlet--multinomial unigram language models.

    Arguments:

    corpus -- grouped corpus
    alpha -- concentration parameter for the Dirichlet prior over theta
    m -- T-dimensional mean of the Dirichlet prior over theta
    beta -- concentration parameter for the Dirichlet prior over phis
    n -- V-dimensional mean of the Dirichlet prior over phis
    """

    D = len(corpus)
    V = len(corpus.vocab)
    T = len(corpus.group_vocab)

    assert len(m) == T and len(n) == V

    Nvt = zeros((V, T), dtype=int)
    Nt = zeros(T, dtype=int)

    Dt = zeros(T, dtype=int)

    for doc, t in corpus:
        Dt[t] += 1
        for v in doc.w:
            Nvt[v, t] += 1
            Nt[t] += 1

    pass # YOUR CODE GOES HERE


def log_evidence_2(corpus, alpha, m, beta, n):

    V = len(corpus.vocab)
    T = len(corpus.group_vocab)

    assert len(m) == T and len(n) == V

    Nvt = zeros((V, T), dtype=int)
    Nt = zeros(T, dtype=int)

    Dt = zeros(T, dtype=int)

    pass # YOUR CODE GOES HERE


def time_taken(func, corpus, alpha, m, beta, n, num_reps):

    avg = 0

    for rep in iterview(xrange(num_reps), inc=1):

        start = time.time()
        func(corpus, alpha, m, beta, n)
        avg += (time.time() - start)

    avg /= float(num_reps)

    return avg


def log_evidence_tokens_1(corpus, beta, n):
    """
    Returns the log evidence for the tokens belonging to a grouped
    corpus given the doucument groups according to a mixture of
    Dirichlet--multinomial unigram language models.

    Arguments:

    corpus -- grouped corpus

    beta -- concentration parameter for the Dirichlet prior over phis
    n -- V-dimensional mean of the Dirichlet prior over phis
    """

    V = len(corpus.vocab)
    T = len(corpus.group_vocab)

    assert len(n) == V

    Nvt = zeros((V, T), dtype=int)
    Nt = zeros(T, dtype=int)

    for doc, t in corpus:
        for v in doc.w:
            Nvt[v, t] += 1
            Nt[t] += 1

    pass # YOUR CODE GOES HERE


def posterior_mean(corpus, alpha, m, beta, n):
    """
    Returns the mean of the posterior distribution.

    Arguments:

    corpus -- grouped corpus
    alpha -- concentration parameter for the Dirichlet prior over theta
    m -- T-dimensional mean of the Dirichlet prior over theta
    beta -- concentration parameter for the Dirichlet prior over phis
    n -- V-dimensional mean of the Dirichlet prior over phis
    """

    mean_theta = posterior_mean_theta(corpus, alpha, m)
    mean_phis = posterior_mean_phis(corpus, beta, n)

    return mean_theta, mean_phis


def posterior_mean_phis(corpus, beta, n):
    """
    Returns the mean of the posterior distribution over phis.

    Arguments:

    corpus -- grouped corpus
    beta -- concentration parameter for the Dirichlet prior over phis
    n -- V-dimensional mean of the Dirichlet prior over phis
    """

    V = len(corpus.vocab)
    T = len(corpus.group_vocab)

    assert len(n) == V

    Nvt = zeros((V, T), dtype=int)
    Nt = zeros(T, dtype=int)

    for doc, t in corpus:
        for v in doc.w:
            Nvt[v, t] += 1
            Nt[t] += 1

    pass # YOUR CODE GOES HERE


def posterior_mean_theta(corpus, alpha, m):
    """
    Returns the mean of the posterior distribution over theta.

    Arguments:

    corpus -- grouped corpus
    alpha -- concentration parameter for the Dirichlet prior over theta
    m -- T-dimensional mean of the Dirichlet prior over theta
    """

    D = len(corpus)
    T = len(corpus.group_vocab)

    assert len(m) == T

    Dt = zeros(T, dtype=int)

    for doc, t in corpus:
        Dt[t] += 1

    pass # YOUR CODE GOES HERE


def print_top_types(corpus, beta, n, num=10):
    """
    Prints the most probable word types according to the mean of the
    posterior distribution over phis.

    Arguments:

    corpus -- grouped corpus
    beta -- concentration parameter for the Dirichlet prior over phis
    n -- V-dimensional mean of the Dirichlet prior over phis

    Keyword arguments:

    num -- number of types to print
    """

    mean_phis = posterior_mean_phis(corpus, beta, n)

    for t in xrange(len(corpus.group_vocab)):
        group = corpus.group_vocab.lookup(t)
        top_types = map(corpus.vocab.lookup, argsort(mean_phis[:, t]))
        print '%s: %s' % (group, ' '.join(top_types[-num:][::-1]))
