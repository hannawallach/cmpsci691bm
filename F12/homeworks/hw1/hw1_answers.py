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

    empirical = zeros(len(dist))

    for n in iterview(xrange(100000)):
        empirical[func(dist)] += 1

    empirical /= num_samples
    normalized_dist = dist / float(dist.sum())

    # could look at max relative error
    # could also look at JS or KL divergence
    # could do absolute difference
    # ...

    error = (abs(empirical - normalized_dist) / normalized_dist).mean()

    assert error < 0.01, 'Mean relative error >= 1%'


def sample_2(dist):

    r = uniform() * dist.sum()

    for x, p in enumerate(dist):
        r -= p
        if r < 0:
            return x


def sample_3(dist):

    cdf = cumsum(dist)
    r = uniform() * cdf[-1]

    for x, acc in enumerate(cdf):
        if acc > r:
            return x


def sample_4(dist):

    cdf = cumsum(dist)
    r = uniform() * cdf[-1]

    return bisect_right(cdf, r)


def sample_5(dist):

    cdf = cumsum(dist)
    r = uniform() * cdf[-1]

    return cdf.searchsorted(r)


def sample_6(dist):

    return multinomial(1, dist / dist.sum()).argmax()


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

    samples = zeros(num_samples)

    cdf = cumsum(dist)
    r = uniform(size=num_samples) * cdf[-1]

    for n in xrange(num_samples):
        samples[n] = cdf.searchsorted(r[n])

    return samples


def sample_9(dist, num_samples=1):

    cdf = cumsum(dist)
    r = uniform(size=num_samples) * cdf[-1]

    return cdf.searchsorted(r)


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

    If the elements of x are log probabilities, they should not be
    exponentiated directly because of underflow. The ratio exp(x[i]) /
    exp(x[j]) = exp(x[i] - x[j]) is not susceptible to underflow,
    however. For any scalar m, log(sum(exp(x))) = log(sum(exp(x) *
    exp(m) / exp(m))) = log(sum(exp(x - m) * exp(m)) = log(exp(m) *
    sum(exp(x - m))) = m + log(sum(exp(x - m))). If m is some element
    of x, this expression involves only ratios of the form exp(x[i]) /
    exp(x[j]) as desired. Setting m = max(x) reduces underflow, while
    avoiding overflow: max(x) is shifted to zero, while all other
    elements of x remain negative, but less so than before. Even in
    the worst case scenario, where exp(x - max(x)) results in
    underflow for the other elements of x, max(x) will be
    returned. Since sum(exp(x)) is dominated by exp(max(x)), max(x) is
    a reasonable approximation to log(sum(exp(x))).
    """

    m = x.max()

    return m + log((exp(x - m)).sum())


def log_sample_1(log_dist):

    r = log(uniform()) + log_sum_exp(log_dist)
    acc = -inf

    for x, l in enumerate(log_dist):
        acc = log_sum_exp(array([acc, l]))
        if acc > r:
            return x


def log_sample_5(log_dist):

    return sample_5(exp(log_dist - log_dist.max()))
