import argparse
import csv
import numpy as np
from collections import Counter


##### Global Constants #####

relations = {'Temporal': 0, 'Contingency': 1, 'Comparison':2, 'Expansion':3,
        'All': 4}
inv_relations = ['Temporal', 'Contingency', 'Comparison', 'Expansion', 'All']

############################


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='train.seg here.')
    parser.add_argument('--test', help='test.seg here.')
    parser.add_argument('--output', help='output file name here.')

    return parser.parse_args()


def ReadTrain(filename):
    x = []
    y = []

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        for line in reader:
            _, sent1, sent2, label = line
            sents = [sent1.split(), sent2.split()]
            x.append(sents)
            y.append(label)

    return x, y


def ReadTest(filename):
    x = []
    IDs = []

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        for line in reader:
            ID, sent1, sent2 = line
            sents = [sent1.split(), sent2.split()]
            x.append(sents)
            IDs.append(ID)

    return x, IDs


def ComputeFreq(x, y):
    cnters = [Counter(), Counter(), Counter(), Counter(), Counter()]

    for i, sents in enumerate(x):
        words = np.unique(sents[0] + sents[1])
        cnters[relations['All']].update(words)
        cnters[relations[y[i]]].update(words)
                
    return cnters


def ComputeProb(cnters, ytrain):
    max_prob = {}
    
    normalize = np.zeros(4)
    for i in range(4):
        normalize[i] = sum(np.array(ytrain) == inv_relations[i])

    for (word, freq) in cnters[relations['All']].most_common():
        if freq < 20:
            break

        probs = np.zeros(4)
        for i in range(4):
            if word in cnters[i]:
                probs[i] = cnters[i][word]/normalize[i]

        probs = probs/sum(probs)
        max_prob[word] = (inv_relations[np.argmax(probs)], max(probs))
            
    return max_prob


def ComputeResult(x, best_relation):
    result = ['Unknown'] * len(x)
    for i, sents in enumerate(x):
        p = 0
        merged_sent = sents[0] + sents[1]

        for word in merged_sent:
            if word in best_relation and best_relation[word][1] > p:
                p = best_relation[word][1]
                result[i] = best_relation[word][0]

    return result


def WriteResult(ID, y, filename):
    N = len(y)
    ID = np.array(ID)
    y = np.array(y)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Relation'])
        writer.writerows(np.append(ID.reshape(N,1), y.reshape(N,1), axis=1))

    return


if __name__ == '__main__':
    args = parse_args()
    
    xtrain, ytrain = ReadTrain(args.train)
    cnters = ComputeFreq(xtrain, ytrain)
    best_relation = ComputeProb(cnters, ytrain)

    xtest, ID = ReadTest(args.test)
    result = ComputeResult(xtest, best_relation)
    WriteResult(ID, result, args.output)
    
