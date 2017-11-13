'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017

This file was used to plot the large amount of data produced through experimentation.
'''

import matplotlib.pyplot as plt
import pandas


# Read in a CSV created by a process
def data_reader(filename):
    base_path = 'Results\\1 HL, 100HN, 10k Gen & Iter\\'

    # Return a transposed dataframe
    return pandas.read_csv(base_path + filename + '.csv', header=None).T


def main():
    # All algorithms and datasets used for filenames
    algorithms = ['BP', 'DE', 'ES', 'GA']
    datasets = ['airfoil', 'concrete', 'forestfires', 'machine', 'yacht']

    # Create a large figure with a separate plot for each dataset
    plt.figure(figsize=(25.5, 13.5), dpi=100)
    subplot = 321

    # Go through all combinations
    for data in datasets:
        for alg in algorithms:
            # Get the data
            temp_df = data_reader(alg + ' ' + data)
            # Plot it on the subplot
            plt.subplot(subplot)
            plt.plot(temp_df, label=alg)

            # Set parameters for this subplot
            plt.title(data + ' Dataset')
            plt.xlabel('Generation')
            plt.ylabel('Mean Squared Error')
            plt.yscale('log')
            plt.xlim(-100, 10000)
            plt.ylim(0, 100000)

        # Move to the next subplot
        subplot += 1

    # Create the legend and fix the layout
    plt.figlegend(loc='lower right')
    plt.tight_layout()
    # Save a PNG of the graph and then show it
    plt.savefig('1 HL, 100HN, 10k Gen.png')
    plt.show()


if __name__ == '__main__':
    main()
