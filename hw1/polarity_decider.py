import argparse
import os
import csv
import numpy as np
from Reader import Reader as review_reader
from collections import Counter
from scipy.sparse import csr_matrix
from sklearn import svm
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-s', '--per_sentence', action='store_true', 
            help='Run testing per sentence.')

    parser.add_argument('-p', '--polarity_file', help='Should give polarity file here.')
    parser.add_argument('-t', '--test_file', help='Should give test review file here.')
    parser.add_argument('-q', '--question_file', help='Should give question file here.')
    parser.add_argument('-o', '--output_file', help='Should give output file here.')
    parser.add_argument('-d', '--NTUSD_path', help='Should put NTUSD\'s path here.')

    parser.add_argument('--train', action='store_true', help='Run training.')
    parser.add_argument('--test', action='store_true', help='Run testing.')

    return parser.parse_args()


def ReadPolarity(filename):
    y = []
    x = []

    with open(filename, 'r') as f:
        for line in f.readlines():
            token = line.split()
            y.append(int(token[0]))
            x.append(token[1:])

    return np.array(y), np.array(x)


def ReadTestReview(filename):
    ID = {}
    feat = []

    with open(filename, 'r') as f:
        content = f.readlines()
        for idx, i in enumerate(np.arange(0, len(content), 2)):
            ID[int(content[i])] = idx
            feat.append(content[i+1].split())
    
    return ID, feat


def ReadQuestion(filename):
    Q = []

    with open(filename, 'r') as f:
        reader = csv.reader(f, delimiter=',')
        next(reader, None)

        for line in reader:
            Q.append((int(line[1]), line[2]))

    return Q


def ReadNTUSD(pathname):
    NTUSD_pos = []
    NTUSD_neg = []

    with open(os.path.join(pathname, 'NTUSD_pos.txt'), 'r') as f:
        for sent in f.readlines():
            NTUSD_pos.append(sent[:-1])
    
    with open(os.path.join(pathname, 'NTUSD_neg.txt'), 'r') as f:
        for sent in f.readlines():
            NTUSD_neg.append(sent[:-1])

    return NTUSD_pos, NTUSD_neg


def WriteDict(dictionary, filename):
    with open(filename, 'w') as f:
        for key in dictionary:
            f.write(key + ' ' + str(dictionary[key]) + '\n')
    return


def ReadDict(filename):
    dictionary = {}

    with open(filename, 'r') as f:
        for line in f.readlines():
            token = line.split()
            key = token[0]
            value = token[1]
            dictionary[key] = value

    return dictionary


def WriteSparseMatrix(filename, sparse_matrix):
    if filename[-4:] == '.npy':
        filename = filename[:-4]

    rows = sparse_matrix[0]
    cols = sparse_matrix[1]
    vals = sparse_matrix[2]
    shape = sparse_matrix[3]
    np.save(filename + '.rows.npy', rows)
    np.save(filename + '.cols.npy', cols)
    np.save(filename + '.vals.npy', vals)
    np.save(filename + '.shape.npy', shape)

    return


def ReadSparseMatrix(filename):
    if filename[-4:] == '.npy':
        filename = filename[:-4]

    rows = np.load(filename + '.rows.npy')
    cols = np.load(filename + '.cols.npy')
    vals = np.load(filename + '.vals.npy')
    shape = np.load(filename + '.shape.npy')

    return (rows, cols, vals, shape)


def WriteResult(filename, y):
    N = len(y)
    y = np.array(y)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Id', 'Label'])
        writer.writerows(np.append(
            np.expand_dims(np.arange(N)+1, axis=1), 
            np.expand_dims(y, axis=1), 
            axis=1))

    return


def WriteResult_per_sent(filename, ID, y):
    N = len(y)
    y = np.array(y)

    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows(np.append(
            np.expand_dims(ID, axis=1),
            np.expand_dims(y, axis=1),
            axis=1))

    return


def ComputeSO(y, x, NTUSD_pos, NTUSD_neg):
    pos_cnter = Counter()
    neg_cnter = Counter()
    all_cnter = Counter()

    for i in range(len(y)):
        if y[i] > 0:
            pos_cnter.update(x[i])
            all_cnter.update(x[i])
        else:
            neg_cnter.update(x[i])
            all_cnter.update(x[i])

    SO_dict = Counter()

    for token in pos_cnter:
        PSO = (pos_cnter[token] + 1) / all_cnter[token]
        NSO = (neg_cnter[token] + 1) / all_cnter[token]
        SO = np.log(PSO/NSO)
        SO_dict[token] = abs(SO)

    for token in neg_cnter:
        if token in SO_dict:
            continue
        PSO = 1 / all_cnter[token]
        NSO = (neg_cnter[token] + 1) / all_cnter[token]
        SO = np.log(PSO/NSO)
        SO_dict[token] = abs(SO)

    k = 2
    hit = 0
    nhit = 0
    for key in SO_dict.keys():
        if key in NTUSD_pos and SO_dict[key] > 0:
            hit += 1
            SO_dict[key] = k * SO_dict[key] + 1
        elif key in NTUSD_neg and SO_dict[key] < 0:
            hit += 1
            SO_dict[key] = k * SO_dict[key] + 1
        else:
            nhit += 1
    print('hit:', hit)
    print('nhit:', nhit)

    return SO_dict


def ComputeFreq(x):
    cnter = Counter()

    for tokens in x:
        cnter.update(tokens)

    return cnter


def build_feat_space(x, SO_dict, freq_cnter, SO_threshold, freq_threshold):
    feat = {}
    ID = 0
    for (key, value) in SO_dict.most_common():
        if value < SO_threshold:
            break
        if freq_cnter[key] >= freq_threshold:
            feat[key] = ID
            ID += 1

    return feat


def word_embedding(x, feat_space, SO):
    D = len(x)
    vocab_size = len(feat_space)
    print('review size:', D)
    print('embedding vocab size:', vocab_size)
    DF = Counter()
    
    rows = []
    cols = []
    vals = []
    shape = [D, vocab_size]

    for sent in x:
        unique_token = np.unique(sent)
        DF.update(unique_token)

    for i in range(D):
        cnter = Counter()
        cnter.update(x[i])
        for element in cnter:
            if element in feat_space:
                TF = cnter[element] / len(x[i])
                TFIDF = TF * np.log(D/DF[element])
                TFSOIDF = TFIDF * (SO[element] + 1)
                rows.append(i)
                cols.append(feat_space[element])
                vals.append(TFSOIDF)

    return (rows, cols, vals, shape)


def run_train(y, sparse_x):
    x_rows = sparse_x[0]
    x_cols = sparse_x[1]
    x_vals = sparse_x[2]
    x_shape = sparse_x[3]
    x = csr_matrix((x_vals, (x_rows, x_cols)), shape=x_shape)

    model = svm.LinearSVC()
    
    print('Start training.')
    model.fit(x, y)
    print('End training')
    pred = model.predict(x)
    print('Training accuracy:', accuracy_score(y, pred))

    return model


def run_test(sparse_x, model):
    x_rows = sparse_x[0]
    x_cols = sparse_x[1]
    x_vals = sparse_x[2]
    x_shape = sparse_x[3]
    x = csr_matrix((x_vals, (x_rows, x_cols)), shape=x_shape)

    pred = model.predict(x)
    return pred


def train(polarity_file, NTUSD_path):
    ytrain, xtrain = ReadPolarity(polarity_file)
    NTUSD_pos, NTUSD_neg = ReadNTUSD(NTUSD_path)

    SO_dict = ComputeSO(ytrain, xtrain, NTUSD_pos, NTUSD_neg)
    freq_cnter = ComputeFreq(xtrain)

    feat_space = build_feat_space(xtrain, SO_dict, freq_cnter, 
            SO_threshold=0.5, 
            freq_threshold=100)
    WriteDict(feat_space, 'polarity_feat.txt')

    new_SO_dict = {}
    for token in feat_space:
        new_SO_dict[token] = SO_dict[token]
    WriteDict(new_SO_dict, 'so_dict.txt')

    xtrain_embed = word_embedding(xtrain, feat_space, new_SO_dict)
    WriteSparseMatrix('polarity_embed.npy', xtrain_embed)

    model = run_train(ytrain, xtrain_embed)
    joblib.dump(model, 'polarity_model.pkl')
    
    return


def test(test_file, question_file, output_file):
    testID, xtest = ReadTestReview(test_file)
    question = ReadQuestion(question_file)

    feat_space = ReadDict('polarity_feat.txt')
    for key in feat_space:
        feat_space[key] = int(feat_space[key])

    SO_dict = ReadDict('so_dict.txt')
    for key in SO_dict:
        SO_dict[key] = float(SO_dict[key])

    xtest_embed = word_embedding(xtest, feat_space, SO_dict)

    model = joblib.load('polarity_model.pkl')
    pred = run_test(xtest_embed, model)

    result = []
    for (ID, aspect) in question:
        result.append(pred[testID[ID]])

    WriteResult(output_file, result)

    return


def test_per_sent(test_file, output_file):
    testID, xtest = [], []
    test_reviews = review_reader.test(test_file)
    for test_review in test_reviews:
        for sent in test_review[1]:
            testID.append(test_review[0])
            xtest.append(sent)

    feat_space = ReadDict('polarity_feat.txt')
    for key in feat_space:
        feat_space[key] = int(feat_space[key])

    SO_dict = ReadDict('so_dict.txt')
    for key in SO_dict:
        SO_dict[key] = float(SO_dict[key])

    xtest_embed = word_embedding(xtest, feat_space, SO_dict)

    model = joblib.load('polarity_model.pkl')
    pred = run_test(xtest_embed, model)

    WriteResult_per_sent(output_file, testID, pred)


if __name__ == '__main__':
    args = parse_args()
    if args.train:
        train(args.polarity_file, args.NTUSD_path)
    if args.test:
        if args.per_sentence:
            test_per_sent(args.test_file, args.output_file)
        else:
            test(args.test_file, args.question_file, args.output_file)

