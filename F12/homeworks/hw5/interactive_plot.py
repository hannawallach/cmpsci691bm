from pylab import clf, draw, ion, plot, xlabel, ylabel


class InteractivePlot(object):

    def __init__(self, xlabel, ylabel):

        self.xlabel = xlabel
        self.ylabel = ylabel

        self.x = []
        self.y = []

        ion()

    def update_plot(self, xval, yval):

        x, y = self.x, self.y

        x.append(xval)
        y.append(yval)

        clf()

        plot(x, y, 'k')

        xlabel(self.xlabel)
        ylabel(self.ylabel)

        draw()


if __name__ == '__main__':

    import time, math

    plt = InteractivePlot('x', 'y')

    for x in xrange(100):

        plt.update_plot(x, x * math.sin(0.4 * x))
        time.sleep(.01)
