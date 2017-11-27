import csv
import scipy.integrate
import scipy.stats
import numpy
import math
import matplotlib.pyplot as plt

data = []
ALPHA = 0.05
MU = 8.8
SIGMA = math.sqrt(19.8)


def reader_csv():
    with open('Analytic.csv') as file_csv:
        reader = csv.DictReader(file_csv)
        for row in reader:
            data.append(int(row['AT']) / 10)


def first_task(mean, var, len_all):
    t0 = ((mean - MU) / var) * math.sqrt(len_all)
    P = scipy.integrate.quad((lambda x: scipy.stats.t.pdf(x, len_all - 1)), t0, numpy.inf)[0]
    print(P)
    print(2 * P > ALPHA == 0)
    return t0


def create_graph(mean, var, len_all):
    x1 = numpy.linspace(-10, 10, 1000)
    x2 = numpy.linspace(100, 300, 1000)

    t0 = first_task(mean, var, len_all)
    chi0 = second_task(mean, var, len_all)

    plt.subplot(211)
    plt.plot(x1, scipy.stats.t.pdf(x1, len_all - 1), color='r')
    plt.plot(t0, scipy.stats.t.pdf(t0, len_all - 1), 'bo', color='b')

    plt.subplot(212)
    plt.plot(x2, scipy.stats.chi2.pdf(x2, len_all - 1), color='r')
    plt.plot(chi0, scipy.stats.chi2.pdf(chi0, len_all - 1), 'bo',color='b')

    plt.show()


def second_task(mean, var, len_all):
    chi0 = ((len_all - 1) * var) / SIGMA ** 2
    print(chi0)
    prob1 = scipy.integrate.quad((lambda x: scipy.stats.chi2.pdf(x, len_all - 1)), -numpy.inf, chi0)[0]
    prob2 = scipy.integrate.quad((lambda x: scipy.stats.chi2.pdf(x, len_all - 1)), chi0, numpy.inf)[0]
    print(prob1, "  ", prob2)
    return chi0


def main():
    reader_csv()
    mean = numpy.mean(data)
    var = numpy.var(data)
    # first_task(mean, var, len(data))
    # second_task(mean, var, len(data))
    create_graph(mean, var, len(data))
    print(mean, " ", var)


if __name__ == '__main__':
    main()
