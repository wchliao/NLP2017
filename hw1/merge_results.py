import argparse
import csv
import numpy as np
from collections import Counter
import random


##### Global Constants #####

aspects_dict = {'服務': 0, '環境': 1, '價格': 2, '交通': 3, '餐廳': 4}

############################


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-s', '--per_sentence', action='store_true', 
            help='Merge per-sentence results.')

    parser.add_argument('-a', '--aspect_file', help='Should give aspect result here.')
    parser.add_argument('-p', '--polarity_file', help='Should give polarity result here.')
    parser.add_argument('-q', '--question_file', help='Should give question file here.')

    return parser.parse_args()


def ReadQuestion(filename):

    Q = []

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        for line in reader:
            Q.append((int(line[1]), line[2]))

    return Q


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


def ReadResult_per_sent(filename):
    ID = []
    result = []

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        
        for line in reader:
            ID.append(int(line[0]))
            result.append(int(line[1]))

    return ID, result


def MergeResult(aspect, polarity):
    N = len(polarity)
    result = []

    for i in range(N):
        if aspect[i] == 0:
            result.append(0)
        else:
            result.append(polarity[i])

    return result


def MergeResult_per_sent(ID, aspect, polarity, question):
    result_dict = {}
    result_update = {}
    for i in np.unique(ID):
        result_dict[i] = [0] * 5
        result_update[i] = [False] * 5

    for i in range(len(ID)):
        if aspect[i] is not 0:
            result_dict[ID[i]][aspect[i]-1] += polarity[i]
            result_update[ID[i]][aspect[i]-1] = True

    result = []

    for (QID, Qaspect) in question:
        score = result_dict[QID][aspects_dict[Qaspect]]
        if score > 0:
            result.append(1)
        elif score < 0:
            result.append(-1)
        elif result_update[QID][aspects_dict[Qaspect]]:
            result.append(random.choice([-1, 1]))
        else:
            result.append(0)


    return result


def main(aspect_file, polarity_file):
    aspect = ReadResult(aspect_file)
    polarity = ReadResult(polarity_file)
    result = MergeResult(aspect, polarity)
    WriteResult('result.csv', result)
    return


def main_per_sent(aspect_file, polarity_file, question_file):
    ID, aspect = ReadResult_per_sent(aspect_file)
    _, polarity = ReadResult_per_sent(polarity_file)
    question = ReadQuestion(question_file)
    result = MergeResult_per_sent(ID, aspect, polarity, question)
    WriteResult('result.csv', result)
    return


if __name__ == '__main__':
    args = parse_args()
    if args.per_sentence:
        main_per_sent(args.aspect_file, args.polarity_file, args.question_file)
    else:
        main(args.aspect_file, args.polarity_file)

