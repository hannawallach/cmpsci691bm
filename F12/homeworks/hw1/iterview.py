import sys, time

__all__ = ['iterview']

WIDTH = 70 # maximum is 78


def progress(n, length):
    """
    Returns a string indicating current progress.
    """

    return '%5.1f%% (%*d/%d)' % ((float(n) / length) * 100, len(str(length)), n, length)


def progress_bar(max_width, n, length):
    """
    Returns a progress bar (string).

    Arguments:

    max_width -- maximum width of the progress bar
    """

    width = int((float(n) / length) * max_width + 0.5) # at least one

    if max_width - width:
        spacing = '>' + (' ' * (max_width - width))[1:]
    else:
        spacing = ''

    return '[%s%s]' % ('=' * width, spacing)


def time_remaining(elapsed, n, length):
    """
    Returns a string indicating the time remaining (if not complete)
    or the total time elapsed (if complete).
    """

    if n == 0:
        return '--:--:--'

    if n == length:
        seconds = int(elapsed) # if complete, total time elapsed
    else:
        seconds = int((elapsed / n) * (length - n)) # otherwise, time remaining

    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)

    return '%02d:%02d:%02d' % (hours, minutes, seconds)


def format(start, n, length):
    """
    returns ...
    """

    string = progress(n, length) + ' ' # current progress

    if n == length:
        end = ' '
    else:
        end = ' ETA '

    end += time_remaining(time.time() - start, n, length) # time remaining (if not complete) or total time elapsed (if complete)

    string += progress_bar(WIDTH - len(string) - len(end), n, length)
    string += end

    return string


def iterview(x, inc=10, length=None):
    """
    Returns an iterator which prints its progress to stderr.

    Arguments:

    x -- iterator
    inc -- number of iterations between printing progress
    length -- hint about the length of x
    """

    start = time.time()
    length = length or len(x)

    if length == 0:
        raise StopIteration

    for n, y in enumerate(x):
        if inc is None or n % inc == 0:
            sys.stderr.write('\r' + format(start, n, length))

        yield y

    sys.stderr.write('\r' + format(start, n+1, length) + '\n')


if __name__ == '__main__':

    for _ in iterview(xrange(400), inc=20):
        time.sleep(0.01)
