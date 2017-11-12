import pandas
import matplotlib.pyplot as plt
import numpy as np


def data_reader(filename):
    base_path = 'Results\\1 HL, 100HN, 10k Gen & Iter\\'

    return pandas.read_csv(base_path + filename + '.csv', header=None).T


def get_filenames():
    algorithms = ['BP', 'DE', 'ES', 'GA']
    #datasets = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']
    #algorithms = ['BP', 'DE']
    datasets = ['airfoil']
    combined = []

    for alg in algorithms:
        for data in datasets:
            combined.append(alg + ' ' + data)

    return combined


def main():
    #files = get_filenames()

    algorithms = ['BP', 'DE', 'ES', 'GA']
    datasets = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']

    np_arrays = []
    thing = {}

    plt.figure(figsize=(25.5, 13.5), dpi=100)
    subplot = 321

    for data in datasets:
        for alg in algorithms:
            temp_df = data_reader(alg + ' ' + data)
            plt.subplot(subplot)
            plt.plot(temp_df, label=alg)
            plt.title(data + ' Dataset')
            plt.xlabel('Generation')
            plt.ylabel('Mean Squared Error')
            plt.yscale('log')
            plt.xlim(-100, 10000)
            plt.ylim(0, 100000)
            #plt.legend()
        subplot += 1

    plt.figlegend(loc='lower right')
    plt.tight_layout()
    plt.savefig('1 HL, 100HN, 10k Gen.png')
    plt.show()


if __name__ == '__main__':
    main()
