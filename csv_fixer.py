'''
CSCI 447: Project 3
Group 28: Trent Baker, Logan Bonney, Bradley White
November 13, 2017

This file contains the functionality to remove a few unnecessary columns from two datasets.
'''

import pandas


def main():
    ff_path = "datasets\\forestfires-old.csv"
    machine_path = 'datasets\\machine-old.csv'
    ff_out = 'datasets\\converted\\forestfires.csv'
    machine_out = 'datasets\\converted\\machine-fixed.csv'

    # Remove Month and Day from Forest Fires. Numbers could be used be that introduces structure and bias to the data
    df = pandas.read_csv(ff_path, header=None)
    df = df.drop([2], axis=1)
    df = df.drop([3], axis=1)
    df.to_csv(ff_out, index=False, header=False)

    # Remove company and model number from machine hardware. They serve no purpose in function approximation.
    # Estimated Relative Performance was removed because we want to estimate the published performance, not a linear
    # regression estimate from previous work.
    df = pandas.read_csv(machine_path, header=None)
    df = df.drop([0], axis=1)
    df = df.drop([1], axis=1)
    df = df.drop([9], axis=1)
    df.to_csv(machine_out, index=False, header=False)


if __name__ == '__main__':
    main()
