from numpy import cumsum, exp, log, searchsorted
from numpy.random import uniform


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


def log_sample(log_dist, num_samples=1):

    return sample(exp(log_dist - log_dist.max()), num_samples)
