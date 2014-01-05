from math import ceil
from numpy import array, cumsum, linspace, searchsorted
from numpy.random import beta, uniform
from pylab import axis, figure, fill_between, grid, show, subplot
from scipy.special import gamma


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


def beta_pdf(x, b, n):
    """
    Arguments:

    x -- list of values for which to return PDF values
    b -- concentration parameter
    n -- mean
    """

    pass # YOUR CODE GOES HERE


def plot_beta_priors(b_values, n_values):
    """
    Arguments:

    b_values -- list of concentration parameters
    n_values -- list of means
    """

    figure().subplots_adjust(left=0.1, wspace=0.35)

    x = linspace(0.01, 0.99, 99)

    subplot_num = 1

    for b in b_values:

        y_values = [beta_pdf(x, b, n) for n in n_values]
        ymax = ceil(array(y_values).max())

        for y in y_values:

            axes = subplot(len(b_values), len(n_values), subplot_num)
            fill_between(x, 0, y, color='gray')

            axis([0.0, 1.0, 0.0, ymax])

            if subplot_num > (len(b_values) - 1) * len(n_values):
                axes.get_xaxis().set_ticks([0.0, 0.5, 1.0])
            else:
                axes.get_xaxis().set_visible(False)

            axes.get_yaxis().set_visible(False)

            grid(False)

            subplot_num += 1

    show()


def generate_corpus(b, n, N):
    """
    Returns a corpus of tokens drawn from a beta--binomial unigram
    language model. Each token is an instance of one of two unique
    word types, represented by vocabulary indices 0 and 1.

    Arguments:

    b -- concentration parameter for the beta prior over phi
    n -- mean of the beta prior over phi
    N -- number of tokens to generate
    """

    pass # YOUR CODE GOES HERE
