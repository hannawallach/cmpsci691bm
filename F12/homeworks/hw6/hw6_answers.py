import cPickle as pickle
from numpy import argsort, exp, log, ones, tile, zeros
from numpy.random import seed
from scipy.special import gammaln

from interactive_plot import InteractivePlot
from iterview import iterview
from preprocess import preprocess
from sample import log_sample, sample


def log_sum_exp(x):

    m = x.max()

    return m + log((exp(x - m)).sum())


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
        self.T = T = len(m) # number of model components

        assert len(n) == V

        # precompute the product of the concentration parameter and
        # mean of the Dirichlet prior over theta

        self.alpha = alpha
        self.alpha_m = alpha * m

        self.gammaln_alpha = gammaln(alpha)
        self.sum_gammaln_alpha_m = gammaln(self.alpha_m).sum()

        # precompute the product of the concentration parameter and
        # mean of the Dirichlet prior over phis

        self.beta = beta
        self.beta_n = beta * n

        self.T_gammaln_beta = T * gammaln(beta)
        self.T_sum_gammaln_beta_n = T * gammaln(self.beta_n).sum()

        # allocate space for N_{v|t} + beta * n_v and N_t + beta

        self.Nvt_plus_beta_n = tile(self.beta_n, (T, 1))
        self.Nt_plus_beta = beta * ones(T)

        self.Dt_plus_alpha_m = self.alpha_m * ones(T)
        self.D_plus_alpha = alpha

        self.z = zeros(D, dtype=int)

    def log_evidence_corpus_and_z(self):
        """
        Returns the log evidence for this instance's corpus (i.e.,
        document tokens) AND current set of document--component
        assignments (i.e., document groups) according to a mixture of
        Dirichlet--multinomial unigram language models.
        """

        D, T = self.D, self.T

        gammaln_alpha = self.gammaln_alpha
        sum_gammaln_alpha_m = self.sum_gammaln_alpha_m

        T_gammaln_beta = self.T_gammaln_beta
        T_sum_gammaln_beta_n = self.T_sum_gammaln_beta_n

        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta

        Dt_plus_alpha_m = self.Dt_plus_alpha_m
        D_plus_alpha = self.D_plus_alpha

        log_e = gammaln_alpha - sum_gammaln_alpha_m + gammaln(Dt_plus_alpha_m).sum() - gammaln(D_plus_alpha) + T_gammaln_beta - T_sum_gammaln_beta_n + gammaln(Nvt_plus_beta_n).sum() - gammaln(Nt_plus_beta).sum()

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

        alpha_m = self.alpha_m

        Dt_plus_alpha_m = self.Dt_plus_alpha_m

        mean_phis = self.posterior_mean_phis()

        for t in xrange(T):
            Dt = Dt_plus_alpha_m[t] - alpha_m[t]
            if Dt > 0:
                top_types = map(corpus.vocab.lookup, argsort(mean_phis[t, :]))
                print 'Component %s (%d documents):' % (t, Dt)
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

        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta

        Dt_plus_alpha_m = self.Dt_plus_alpha_m

        z = self.z

        for d, (doc, t) in enumerate(iterview(zip(corpus, z))):

            if not init:
                Nvt_plus_beta_n[t, :] -= doc.Nv
                Nt_plus_beta[t] -= len(doc)
                Dt_plus_alpha_m[t] -= 1

            t = log_sample(gammaln(Nt_plus_beta) - gammaln(Nvt_plus_beta_n).sum(axis=1) + gammaln(tile(doc.Nv, (T, 1)) + Nvt_plus_beta_n).sum(axis=1) - gammaln(len(doc) * ones(T) + Nt_plus_beta) + log(Dt_plus_alpha_m))

            Nvt_plus_beta_n[t, :] += doc.Nv
            Nt_plus_beta[t] += len(doc)
            Dt_plus_alpha_m[t] += 1

            z[d] = t

    @classmethod
    def load(cls, filename):
        return pickle.load(file(filename, 'r'))

    def save(self, filename):
        pickle.dump(self, file(filename, 'wb'))

    def log_predictive_prob(self, new_corpus, num_samples):

        D, V, T = self.D, self.V, self.T

        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta

        Dt_plus_alpha_m = self.Dt_plus_alpha_m
        D_plus_alpha = self.D_plus_alpha

        Nvt_new, Nt_new, Dt_new, z_new = [], [], [], []

        for r in xrange(num_samples):

            Nvt_new.append(zeros((T, V), dtype=int))
            Nt_new.append(zeros(T, dtype=int))

            Dt_new.append(zeros(T, dtype=int))

            z_new.append(zeros(len(new_corpus), dtype=int))

        log_p = 0

        for d, doc in enumerate(iterview(new_corpus)):

            tmp = zeros(num_samples, dtype=float)

            for r in xrange(num_samples):
                for prev_d in xrange(0, d):

                    prev_doc = corpus.documents[prev_d]
                    t = z_new[r][prev_d]

                    Nvt_new[r][t, :] -= prev_doc.Nv
                    Nt_new[r][t] -= len(prev_doc)
                    Dt_new[r][t] -= 1

                    t = log_sample(gammaln(Nt_new[r] + Nt_plus_beta) - gammaln(Nvt_new[r] + Nvt_plus_beta_n).sum(axis=1) + gammaln(tile(prev_doc.Nv, (T, 1)) + Nvt_new[r] + Nvt_plus_beta_n).sum(axis=1) - gammaln(len(prev_doc) * ones(T) + Nt_new[r] + Nt_plus_beta) + log(Dt_new[r] + Dt_plus_alpha_m))

                    Nvt_new[r][t, :] += prev_doc.Nv
                    Nt_new[r][t] += len(prev_doc)
                    Dt_new[r][t] += 1

                    z_new[r][prev_d] = t

                log_dist = gammaln(Nt_new[r] + Nt_plus_beta) - gammaln(Nvt_new[r] + Nvt_plus_beta_n).sum(axis=1) + gammaln(tile(doc.Nv, (T, 1)) + Nvt_new[r] + Nvt_plus_beta_n).sum(axis=1) - gammaln(len(doc) * ones(T) + Nt_new[r] + Nt_plus_beta) + log(Dt_new[r] + Dt_plus_alpha_m) - log(d + D_plus_alpha)

                tmp[r] = log_sum_exp(log_dist)

                t = log_sample(log_dist)

                Nvt_new[r][t, :] += doc.Nv
                Nt_new[r][t] += len(doc)
                Dt_new[r][t] += 1

                z_new[r][d] = t

            log_p += log_sum_exp(tmp) - log(num_samples)

        return log_p


if __name__ == '__main__':

    extra_stopwords = ['answer', 'dont', 'find', 'im', 'information', 'ive', 'message', 'question', 'read', 'science', 'wondering']

    corpus = preprocess('questions.csv', 'stopwordlist.txt', extra_stopwords)

    train_corpus = corpus[:-100]
    assert train_corpus.vocab == corpus.vocab

    test_corpus = corpus[-100:]
    assert test_corpus.vocab == corpus.vocab

    V = len(corpus.vocab)
    T = 10

    alpha = 0.1 * T
    m = ones(T) / T

    beta = 0.01 * V
    n = ones(V) / V

    mm = MixtureModel(train_corpus, alpha, m, beta, n)
    mm.gibbs(num_itns=25, random_seed=1000)
#    mm.save('model.dat')

#    mm = MixtureModel.load('model.dat')
    print mm.log_predictive_prob(test_corpus, num_samples=5)
