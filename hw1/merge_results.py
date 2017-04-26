import argparse
import csv
import numpy as np
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-a', '--aspect_file', help='Should give aspect result here.')
    parser.add_argument('-p', '--polarity_file', help='Should give polarity result here.')

    return parser.parse_args()


def WriteResult(filename, y):
    N = len(y)
    y = np.array(y)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Label'])
        writer.writerows(np.append(np.arange(N).reshape(N,1) + 1, 
            y.reshape(N,1), axis=1))

    return


def ReadResult(filename):

    result = []

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        for line in reader:
            result.append(int(line[1]))

    return result


def MergeResult(aspect, polarity):
    N = len(polarity)
    result = []

    for i in range(N):
        if aspect[i] == 0:
            result.append(0)
        else:
            result.append(polarity[i])

    return result


def main(aspect_file, polarity_file):
    aspect = ReadResult(aspect_file)
    polarity = ReadResult(polarity_file)
    result = MergeResult(aspect, polarity)
    WriteResult('result.csv', result)
    return


if __name__ == '__main__':
    args = parse_args()
    main(args.aspect_file, args.polarity_file)

