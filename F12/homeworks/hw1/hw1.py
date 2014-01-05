import sys, time
from bisect import bisect_right
from numpy import array, cumsum, exp, inf, log, ones, zeros
from numpy.random import multinomial, uniform, seed
from numpy.random.mtrand import dirichlet
from pylab import figure, legend, plot, show, xlabel, ylabel

from iterview import iterview

def get_dists(dim, num):
    """
    Returns an array of discrete distributions.

    Arguments:

    dim -- dimensionality of the distributions
    num -- number of distributions to return
    """

    return dirichlet(ones(dim), num)


def sample_1(dist):
    """
    Uses the inverse CDF method to return a sample drawn from an
    (unnormalized) discrete distribution.

    Arguments:

    dist -- (unnormalized) distribution
    """

    r = uniform() * dist.sum()
    acc = 0

    for x, p in enumerate(dist):
        acc += p
        if acc > r:
            return x


def check(func, dist):
    """
    Arguments:

    func -- function to check
    dist -- (unnormalized) distribution to pass to func
    """

    pass # YOUR CODE GOES HERE


def sample_2(dist):

    pass # YOUR CODE GOES HERE


def sample_3(dist):

    pass # YOUR CODE GOES HERE


def sample_4(dist):

    pass # YOUR CODE GOES HERE


def sample_5(dist):

    pass # YOUR CODE GOES HERE


def sample_6(dist):

    pass # YOUR CODE GOES HERE


def time_taken(func, dists, num_reps, num_samples=1):

    seed(1000)

    mean = 0

    for rep in iterview(xrange(num_reps)):

        start = time.time()

        for dist in dists:
            if num_samples == 1:
                func(dist)
            else:
                func(dist, num_samples)

        mean += (time.time() - start) / float(len(dists))

    mean /= float(num_reps)

    return mean


def plot_by_dimension(functions, dimensions, log_space=False):
    """
    Arguments:

    functions -- list of functions
    dimensions -- list of dimensionalities

    Keyword arguments:

    log_space -- whether to represent distributions in log space
    """

    num_dists = 100
    num_reps = 50

    dists_by_dimensions = [(d, get_dists(d, num_dists)) for d in dimensions]

    if log_space:
        for i, (d, dist) in enumerate(dists_by_dimensions):
            dists_by_dimensions[i] = (d, log(dist))

    figure()

    for func in functions:

        print 'Function: ' + func.__name__

        times = []

        for d, dists in dists_by_dimensions:
            print 'Dimensionality:', d
            times.append(time_taken(func, dists, num_reps, 1))

        plot(dimensions, times, label=func.__name__)

    legend()

    xlabel('Dimensionality')
    ylabel('Average time to draw one sample')

    show()


def sample_7(dist, num_samples=1):

    samples = zeros(num_samples)

    for n in xrange(num_samples):

        cdf = cumsum(dist)
        r = uniform() * cdf[-1]

        samples[n] = cdf.searchsorted(r)

    return samples


def sample_8(dist, num_samples=1):

    pass # YOUR CODE GOES HERE


def sample_9(dist, num_samples=1):

    pass # YOUR CODE GOES HERE


def plot_by_num_samples(functions, num_samples):
    """
    Arguments:

    functions -- list of functions
    num_samples -- list of numbers of samples
    """

    num_dists = 100
    num_reps = 50

    dists = get_dists(100, num_dists)

    figure()

    for func in functions:

        print 'Function: ' + func.__name__

        times = []

        for n in num_samples:
            print 'Number of samples:', n
            times.append(time_taken(func, dists, num_reps, n))

        plot(num_samples, times, label=func.__name__)

    legend()

    xlabel('Number of samples')
    ylabel('Average time to draw samples')

    show()


def log_sum_exp(x):
    """
    Returns log(sum(exp(x))).
    """

    m = x.max()

    return m + log((exp(x - m)).sum())


def log_sample_1(log_dist):

    pass # YOUR CODE GOES HERE


def log_sample_5(log_dist):

    pass # YOUR CODE GOES HERE
