import cPickle as pickle
from numpy import argsort, array, exp, log, ones, tile, zeros
from numpy.random import seed, uniform
from scipy.special import gammaln, psi

from interactive_plot import InteractivePlot
from iterview import iterview
from preprocess import preprocess
from sample import sample


def log_sum_exp(x):

    m = x.max()

    return m + log((exp(x - m)).sum())


class LDA(object):

    def __init__(self, corpus, alpha, m, beta, n):
        """
        Initializes all data attributes for LDA using the specified
        corpus and hyperparameter values.

        Arguments:

        corpus -- corpus of documents
        alpha -- concentration parameter for the Dirichlet prior over thetas
        m -- T-dimensional mean of the Dirichlet prior over thetas
        beta -- concentration parameter for the Dirichlet prior over phis
        n -- V-dimensional mean of the Dirichlet prior over phis
        """

        self.corpus = corpus

        self.D = D = len(corpus) # number of documents
        self.V = V = len(corpus.vocab) # vocabulary size
        self.T = T = len(m) # number of model components (i.e., topics)

        assert len(n) == V

        # precompute the product of the concentration parameter and
        # mean of the Dirichlet prior over thetas

        self.alpha, self.m = alpha, m
        self.alpha_m = alpha * m

        self.D_gammaln_alpha = D * gammaln(alpha)
        self.D_sum_gammaln_alpha_m = D * gammaln(self.alpha_m).sum()

        # precompute the product of the concentration parameter and
        # mean of the Dirichlet prior over phis

        self.beta, self.n = beta, n
        self.beta_n = beta * n

        self.T_gammaln_beta = T * gammaln(beta)
        self.T_sum_gammaln_beta_n = T * gammaln(self.beta_n).sum()

        # allocate space for N_{v|t} + beta * n_v and N_t + beta

        self.Nvt_plus_beta_n = tile(self.beta_n, (T, 1)).T
        self.Nt_plus_beta = beta * ones(T)

        self.Ntd_plus_alpha_m = tile(self.alpha_m, (D, 1))
        self.Nd_plus_alpha = alpha * ones(D)

        self.z = []

        for doc in corpus:
            self.z.append(zeros(len(doc), dtype=int))

    def log_evidence_corpus_and_z(self, alpha=None, beta=None):
        """
        Returns the log evidence for this instance's corpus (i.e.,
        document tokens) AND current set of token--component (i.e.,
        token--topic) assignments according to LDA.
        """

        D, T = self.D, self.T

        if alpha:

            alpha_m = alpha * self.m

            D_gammaln_alpha = D * gammaln(alpha)
            D_sum_gammaln_alpha_m = D * gammaln(alpha_m).sum()

            Ntd_plus_alpha_m = self.Ntd_plus_alpha_m + tile(alpha_m - self.alpha_m, (D, 1))
            Nd_plus_alpha = self.Nd_plus_alpha + alpha - self.alpha

        else:

            D_gammaln_alpha = self.D_gammaln_alpha
            D_sum_gammaln_alpha_m = self.D_sum_gammaln_alpha_m

            Ntd_plus_alpha_m = self.Ntd_plus_alpha_m
            Nd_plus_alpha = self.Nd_plus_alpha

        if beta:

            beta_n = beta * self.n

            T_gammaln_beta = T * gammaln(beta)
            T_sum_gammaln_beta_n = T * gammaln(beta_n).sum()

            Nvt_plus_beta_n = self.Nvt_plus_beta_n + tile(beta_n - self.beta_n, (T, 1)).T
            Nt_plus_beta = self.Nt_plus_beta + beta - self.beta

        else:

            T_gammaln_beta = self.T_gammaln_beta
            T_sum_gammaln_beta_n = self.T_sum_gammaln_beta_n

            Nvt_plus_beta_n = self.Nvt_plus_beta_n
            Nt_plus_beta = self.Nt_plus_beta

        log_e = D_gammaln_alpha - D_sum_gammaln_alpha_m + gammaln(Ntd_plus_alpha_m).sum() - gammaln(Nd_plus_alpha).sum() + T_gammaln_beta - T_sum_gammaln_beta_n + gammaln(Nvt_plus_beta_n).sum() - gammaln(Nt_plus_beta).sum()

        return log_e

    def posterior_mean_phis(self):
        """
        Returns an approximation of the mean of the posterior
        distribution over phis computed using the current set of
        token--component (i.e., token--topic) assignments
        """

        V, T = self.V, self.T

        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta

        return Nvt_plus_beta_n / tile(Nt_plus_beta, (V, 1))

    def print_top_types(self, num=10):
        """
        Prints the most probable word types according to the current
        approximation of mean of the posterior distribution over phis.

        Keyword arguments:

        num -- number of types to print
        """

        corpus = self.corpus

        T = self.T

        beta = self.beta

        Nt_plus_beta = self.Nt_plus_beta

        mean_phis = self.posterior_mean_phis()

        for t in xrange(T):
            Nt = Nt_plus_beta[t] - beta
            top_types = map(corpus.vocab.lookup, argsort(mean_phis[:, t]))
            print 'Component %s (%d tokens):' % (t, Nt)
            print '* %s' % ' '.join(top_types[-num:][::-1])

    def gibbs(self, num_itns=250, random_seed=None, optimize_alpha_m=False, optimize_beta=False, slice_sample=False):
        """
        Uses Gibbs sampling to draw multiple samples from the
        posterior distribution over token--component (token--topic)
        assignments given this instance's corpus (i.e., document
        tokens). After drawing each sample, the log evidence for this
        instance's corpus AND current set of token--component (i.e.,
        token--topic) assignments is printed, along with the most
        probable word types according to the current approximation of
        mean of the posterior distribution over phis.

        Keyword arguments:

        num_itns -- number of Gibbs sampling iterations
        random_seed -- seed for the random number generator
        optimize_alpha_m -- whether to optimize alpha and m
        optimize_beta -- whether to optimize beta
        slice_sample -- whether to slice sample alpha and beta
        """

        assert not (slice_sample and (optimize_alpha_m or optimize_beta))

        seed(random_seed)

        print 'Initialization:'

        self.gibbs_iteration(init=True)

        log_e = self.log_evidence_corpus_and_z()

        plt = InteractivePlot('Iteration', 'Log Evidence')
        plt.update_plot(0, log_e)

        print '\nLog evidence: %s' % log_e
        print 'alpha, beta: %s, %s\n' % (self.alpha, self.beta)
        self.print_top_types()

        for itn in xrange(1, num_itns + 1):

            print '\n---\n\nIteration %s:' % itn

            self.gibbs_iteration()

            if slice_sample:
                self.slice_sample()

            if optimize_alpha_m:
                self.optimize_alpha_m()

            if optimize_beta:
                self.optimize_beta()

            log_e = self.log_evidence_corpus_and_z()

            plt.update_plot(itn, log_e)

            print '\nLog evidence: %s' % log_e
            print 'alpha, beta: %s, %s\n' % (self.alpha, self.beta)
            self.print_top_types()

    def gibbs_iteration(self, init=False):
        """
        Uses Gibbs sampling to draw a single sample from the posterior
        distribution over token--component (i.e., token--topic)
        assignments given this instance's corpus (i.e., document
        tokens). By default (i.e., if keyword argument 'init' is set
        to the value 'False') all token--component assignments (and
        corresponding counts) are assumed to have been initialized
        previously; otherwise, they are initialized.

        Keyword arguments:

        init -- whether to initialize token--component assignments
        """

        corpus = self.corpus

        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta
        Ntd_plus_alpha_m = self.Ntd_plus_alpha_m
        Nd_plus_alpha = self.Nd_plus_alpha

        z = self.z

        for d, (doc, zd) in enumerate(iterview(zip(corpus, z), inc=200)):
            for n, (v, t) in enumerate(zip(doc.w, zd)):

                if not init:
                    Nvt_plus_beta_n[v, t] -= 1
                    Nt_plus_beta[t] -= 1
                    Ntd_plus_alpha_m[d, t] -= 1

                t = sample((Nvt_plus_beta_n[v, :] / Nt_plus_beta) * Ntd_plus_alpha_m[d, :])

                Nvt_plus_beta_n[v, t] += 1
                Nt_plus_beta[t] += 1
                Ntd_plus_alpha_m[d, t] +=1

                if init:
                    Nd_plus_alpha[d] += 1

                zd[n] = t

    def slice_sample(self, num_itns=5, step_size=1.0):
        """
        Uses slice sampling to draw multiple samples from the
        posterior distribution over concentration parameters alpha and
        beta given this instance's corpus AND the current set of
        token--component (i.e., token--topic) assignments.

        Keyword arguments:

        num_itns -- number of slice sampling iterations
        step_size -- step (slice) size
        """

        x = array([log(self.alpha), log(self.beta)])

        for itn in xrange(1, num_itns + 1):

            a, b = exp(x[0]), exp(x[1])

            log_p = self.log_evidence_corpus_and_z(alpha=a, beta=b) + x.sum()
            new_u = log(uniform()) + log_p

            l = x - uniform(size=2) * step_size
            r = l + step_size

            while True:

                new_x = l + uniform(size=2) * (r - l)

                new_a, new_b = exp(new_x[0]), exp(new_x[1])

                if (self.log_evidence_corpus_and_z(alpha=new_a, beta=new_b) + new_x.sum()) > new_u:
                    break
                else:
                    for i in xrange(2):
                        if new_x[i] < x[i]:
                            l[i] = new_x[i]
                        else:
                            r[i] = new_x[i]

            x = new_x

        alpha, beta = exp(x[0]), exp(x[1])
        alpha_m, beta_n = alpha * self.m, beta * self.n

        D, T = self.D, self.T

        self.D_gammaln_alpha = D * gammaln(alpha)
        self.D_sum_gammaln_alpha_m = D * gammaln(alpha_m).sum()

        self.T_gammaln_beta = T * gammaln(beta)
        self.T_sum_gammaln_beta_n = T * gammaln(beta_n).sum()

        self.Nvt_plus_beta_n += tile(beta_n - self.beta_n, (T, 1)).T
        self.Nt_plus_beta += beta - self.beta

        self.Ntd_plus_alpha_m += tile(alpha_m - self.alpha_m, (D, 1))
        self.Nd_plus_alpha += alpha - self.alpha

        self.alpha, self.alpha_m = alpha, alpha_m
        self.beta, self.beta_n = beta, beta_n

    def optimize_alpha_m(self, num_itns=5):
        """
        Jointly optimizes hyperparameters alpha and m using the
        current set of token--component assignments.

        Keyword arguments:

        num_itns -- number of optimization iterations
        """

        D = self.D

        alpha, alpha_m = self.alpha, self.alpha_m

        Ntd = self.Ntd_plus_alpha_m - tile(alpha_m, (D, 1))
        Nd = self.Nd_plus_alpha - alpha

        new_alpha, new_alpha_m = alpha, alpha_m.copy()

        for itn in xrange(1, num_itns + 1):

            pass # YOUR CODE GOES HERE

            new_alpha = new_alpha_m.sum()

        self.D_gammaln_alpha = D * gammaln(new_alpha)
        self.D_sum_gammaln_alpha_m = D * gammaln(new_alpha_m).sum()

        self.Ntd_plus_alpha_m = Ntd + tile(new_alpha_m, (D, 1))
        self.Nd_plus_alpha = Nd + new_alpha

        self.alpha, self.alpha_m = new_alpha, new_alpha_m

    def optimize_beta(self, num_itns=5):
        """
        Optimizes hyperparameter beta using the current set of
        token--component assignments; n is assumed to be uniform.

        Keyword arguments:

        num_itns -- number of optimization iterations
        """

        T = self.T

        beta, n, beta_n = self.beta, self.n, self.beta_n

        Nvt = self.Nvt_plus_beta_n - tile(beta_n, (T, 1)).T
        Nt = self.Nt_plus_beta - beta

        new_beta, new_beta_n = beta, beta_n.copy()

        for itn in xrange(1, num_itns + 1):

            pass # YOUR CODE GOES HERE

            new_beta_n = new_beta * n

        self.T_gammaln_beta = T * gammaln(new_beta)
        self.T_sum_gammaln_beta_n = T * gammaln(new_beta_n).sum()

        self.Nvt_plus_beta_n = Nvt + tile(new_beta_n, (T, 1)).T
        self.Nt_plus_beta = Nt + new_beta

        self.beta, self.beta_n = new_beta, new_beta_n

    @classmethod
    def load(cls, filename):
        return pickle.load(file(filename, 'r'))

    def save(self, filename):
        pickle.dump(self, file(filename, 'wb'))

    def log_predictive_prob(self, new_corpus, num_samples):
        """
        Returns an approximation of the log probability of the
        specified new corpus given this instance's corpus (i.e.,
        document tokens) AND current set of token--component (i.e.,
        token--topic) assignments according to LDA.

        Arguments:

        new_corpus -- new corpus of documents
        num_samples -- ...
        """

        V, T = self.V, self.T

        D_new = len(new_corpus)

        alpha, alpha_m = self.alpha, self.alpha_m

        Nvt_plus_beta_n = self.Nvt_plus_beta_n
        Nt_plus_beta = self.Nt_plus_beta

        Nvt_new, Nt_new, Ntd_new, z_new = [], [], [], []

        for r in xrange(num_samples):

            Nvt_new.append(zeros((V, T), dtype=int))
            Nt_new.append(zeros(T, dtype=int))
            Ntd_new.append(zeros((D_new, T), dtype=int))

            z_r = []

            for doc in new_corpus:
                z_r.append(zeros(len(doc), dtype=int))

            z_new.append(z_r)

        log_p = 0

        for d, doc in enumerate(iterview(new_corpus)):
            for n, v in enumerate(doc.w):

                tmp = zeros(num_samples, dtype=float)

                for r in xrange(num_samples):

                    # for efficiency, resample only those
                    # token--component assignments belonging to
                    # previous tokens in the current document

                    for prev_n in xrange(0, n):

                        prev_v = doc.w[prev_n]
                        t = z_new[r][d][prev_n]

                        Nvt_new[r][prev_v, t] -= 1
                        Nt_new[r][t] -= 1
                        Ntd_new[r][d, t] -= 1

                        t = sample((Nvt_new[r][prev_v, :] + Nvt_plus_beta_n[prev_v, :]) / (Nt_new[r] + Nt_plus_beta) * (Ntd_new[r][d, :] + alpha_m))

                        Nvt_new[r][prev_v, t] += 1
                        Nt_new[r][t] += 1
                        Ntd_new[r][d, t] += 1

                        z_new[r][d][prev_n] = t

                    dist = ((Nvt_new[r][v, :] + Nvt_plus_beta_n[v, :]) / (Nt_new[r] + Nt_plus_beta)) * ((Ntd_new[r][d, :] + alpha_m) / (n + alpha))

                    tmp[r] = log(dist.sum())

                    t = sample(dist)

                    Nvt_new[r][v, t] += 1
                    Nt_new[r][t] += 1
                    Ntd_new[r][d, t] += 1

                    z_new[r][d][n] = t

                log_p += log_sum_exp(tmp) - log(num_samples)

        return log_p


if __name__ == '__main__':

    extra_stopwords = ['answer', 'dont', 'find', 'im', 'information', 'ive', 'message', 'question', 'read', 'science', 'wondering']

    corpus = preprocess('questions.csv', 'stopwordlist.txt', extra_stopwords)

#    extra_stopwords = ['light', 'lights', 'object', 'objects', 'sky']

#    corpus = preprocess('ufos_small.csv', 'stopwordlist.txt', extra_stopwords)

    train_corpus = corpus[:-100]
    assert train_corpus.vocab == corpus.vocab

    test_corpus = corpus[-100:]
    assert test_corpus.vocab == corpus.vocab

    V = len(corpus.vocab)
    T = 100

    alpha = 0.1 * T
    m = ones(T) / T

    beta = 0.01 * V
    n = ones(V) / V

    lda = LDA(train_corpus, alpha, m, beta, n)
    lda.gibbs(num_itns=250, random_seed=1000)
#    lda.save('model.dat')

#    lda = LDA.load('model.dat')
    print lda.log_predictive_prob(test_corpus, num_samples=5)
