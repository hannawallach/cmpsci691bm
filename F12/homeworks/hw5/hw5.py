from numpy import argsort, log, ones, tile, zeros
from numpy.random import seed
from scipy.special import gammaln

from interactive_plot import InteractivePlot
from iterview import iterview
from preprocess import preprocess
from sample import log_sample, sample


class MixtureModel(object):

    def __init__(self, corpus, alpha, m, beta, n):
        """
        Initializes all data attributes for a mixture of
        Dirichlet--multinomial unigram language models using the
        specified corpus and hyperparameter values.

        Arguments:

        corpus -- corpus of documents
        alpha -- concentration parameter for the Dirichlet prior over theta
        m -- T-dimensional mean of the Dirichlet prior over theta
        beta -- concentration parameter for the Dirichlet prior over phis
        n -- V-dimensional mean of the Dirichlet prior over phis
        """

        self.corpus = corpus

        self.D = D = len(corpus) # number of documents
        self.V = V = len(corpus.vocab) # vocabulary size
        self.T = len(m) # number of model components

        assert len(n) == V

        # precompute the product of the concentration parameter and
        # mean of the Dirichlet prior over theta

        self.alpha = alpha
        self.alpha_m = alpha * m

        # precompute the product of the concentration parameter and
        # mean of the Dirichlet prior over phis

        self.beta = beta
        self.beta_n = beta * n

        # allocate space for N_{v|t} + beta * n_v and N_t + beta

        self.Nvt_plus_beta_n = tile(self.beta_n, (self.T, 1))
        self.Nt_plus_beta = beta * ones(self.T)

        self.Dt = zeros(self.T, dtype=int)

        self.z = zeros(D, dtype=int)

    def log_evidence_corpus_and_z(self):
        """
        Returns the log evidence for this instance's corpus (i.e.,
        document tokens) AND current set of document--component
        assignments (i.e., document groups) according to a mixture of
        Dirichlet--multinomial unigram language models.
        """

        D, T = self.D, self.T

        alpha, alpha_m = self.alpha, self.alpha_m
        beta, beta_n = self.beta, self.beta_n

        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta

        Dt = self.Dt

        log_e = gammaln(alpha) - gammaln(alpha_m).sum() + gammaln(Dt + alpha_m).sum() - gammaln(D + alpha) + T * gammaln(beta) - T * gammaln(beta_n).sum() + gammaln(Nvt_plus_beta_n).sum() - gammaln(Nt_plus_beta).sum()

        return log_e

    def posterior_mean_phis(self):
        """
        Returns an approximation of the mean of the posterior
        distribution over phis computed using the current set of
        document--component assignments (i.e., document groups).
        """

        V, T = self.V, self.T

        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta

        return Nvt_plus_beta_n / tile(Nt_plus_beta, (V, 1)).T

    def print_top_types(self, num=10):
        """
        Prints the most probable word types according to the current
        approximation of mean of the posterior distribution over phis.

        Keyword arguments:

        num -- number of types to print
        """

        corpus = self.corpus

        T = self.T

        Dt = self.Dt

        mean_phis = self.posterior_mean_phis()

        for t in xrange(T):
            if Dt[t] > 0:
                top_types = map(corpus.vocab.lookup, argsort(mean_phis[t, :]))
                print 'Component %s (%s documents):' % (t, Dt[t])
                print '* %s' % ' '.join(top_types[-num:][::-1])

    def gibbs(self, num_itns=25, random_seed=None):
        """
        Uses Gibbs sampling to draw multiple samples from the
        posterior distribution over document--component assignments
        (i.e., document groups) given this instance's corpus (i.e.,
        document tokens). After drawing each sample, the log evidence
        for this instance's corpus AND current set of
        document--component assignments is printed, along with the
        most probable word types according to the current
        approximation of mean of the posterior distribution over phis.

        Keyword arguments:

        num_itns -- number of Gibbs sampling iterations
        random_seed -- seed for the random number generator
        """

        seed(random_seed)

        print 'Initialization:'

        self.gibbs_iteration(init=True)

        log_e = self.log_evidence_corpus_and_z()

        plt = InteractivePlot('Iteration', 'Log Evidence')
        plt.update_plot(0, log_e)

        print '\nLog evidence: %s\n' % log_e
        self.print_top_types()

        for itn in xrange(1, num_itns + 1):

            print '\n---\n\nIteration %s:' % itn

            self.gibbs_iteration()

            log_e = self.log_evidence_corpus_and_z()

            plt.update_plot(itn, log_e)

            print '\nLog evidence: %s\n' % log_e
            self.print_top_types()

    def gibbs_iteration(self, init=False):
        """
        Uses Gibbs sampling to draw a single sample from the posterior
        distribution over document--component assignments (i.e.,
        document groups) given this instance's corpus (i.e., document
        tokens). By default (i.e., if keyword argument 'init' is set
        to the value 'False') all document--component assignments (and
        corresponding counts) are assumed to have been initialized
        previously; otherwise, they are initialized.

        Keyword arguments:

        init -- whether to initialize document--component assignments
        """

        corpus = self.corpus

        T = self.T

        alpha_m = self.alpha_m

        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta

        Dt = self.Dt

        z = self.z

        for d, (doc, t) in enumerate(iterview(zip(corpus, z))):

            if not init:
                Nvt_plus_beta_n[t, :] -= doc.Nv
                Nt_plus_beta[t] -= len(doc)
                Dt[t] -= 1

            pass # YOUR CODE GOES HERE

            Nvt_plus_beta_n[t, :] += doc.Nv
            Nt_plus_beta[t] += len(doc)
            Dt[t] += 1

            z[d] = t


if __name__ == '__main__':

    extra_stopwords = ['answer', 'dont', 'find', 'im', 'information', 'ive', 'message', 'question', 'read', 'science', 'wondering']

    corpus = preprocess('questions.csv', 'stopwordlist.txt', extra_stopwords)

    V = len(corpus.vocab)
    T = 10

    mm = MixtureModel(corpus, 0.1 * T, ones(T) / T, 0.01 * V, ones(V) / V)
    mm.gibbs(num_itns=25)
