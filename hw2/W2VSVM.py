# coding: utf-8

import argparse
import numpy as np
import pandas as pd
from sklearn import svm
import gensim

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--vectorSize', help='vector size of Word2Vec')

    return parser.parse_args()

def balanceData(df):
    print('Before:')
    print(df.Relation.value_counts())
    df_expa = df[df['Relation'] == 0]
    df_cont = df[df['Relation'] == 1]
    df_comp = df[df['Relation'] == 2]
    df_temp = df[df['Relation'] == 3]
    frames = [df_expa.copy()]+[df_cont.copy()]*3+[df_comp.copy()]*5+[df_temp.copy()]*6
    result = pd.concat(frames, ignore_index='true')
    print('After:')
    print(result.Relation.value_counts())
    return result

def generateContents(df_complete, df_test):
    contents = []
    for i in range(df_complete.shape[0]):
        contents.append(df_complete['Clause1'][i].split(' '))
        contents.append(df_complete['Clause2'][i].split(' '))
    for i in range(df_test.shape[0]):
        contents.append(df_test['Clause1'][i].split(' '))
        contents.append(df_test['Clause2'][i].split(' '))
    return contents

def generateX(df, vectorSize):
    X = np.empty(shape=(df.shape[0], vectorSize*2))
    for i in range(df.shape[0]):
        X[i] = np.concatenate((np.sum(np.asarray(model[term]) for term in df['Clause1'][i].split(' ')), np.sum(np.asarray(model[term]) for term in df['Clause2'][i].split(' '))), axis=0)
    return X

if __name__ == '__main__':

    args = parse_args()
    args.vectorSize = int(args.vectorSize)

    numToRelation = ['Expansion', 'Contingency', 'Comparison', 'Temporal']

    # Read complete data
    df_complete = pd.read_csv('./data/train.simp.seg')

    # Read test
    df_test = pd.read_csv('./data/test.simp.seg')

    # Generate contents by train, valid, test
    contents = generateContents(df_complete, df_test)

    # Train Word2Vec model
    model = gensim.models.Word2Vec(contents, size=args.vectorSize, window=5, min_count=0, workers=4)

    ### Cross validation
    accs = []
    for i in range(5):
        print('Fold', i, '....')
        df_train = pd.read_csv('./myData/train'+str(i)+'.csv').replace('Expansion', 0).replace('Contingency', 1).replace('Comparison', 2).replace('Temporal', 3).drop('Id', 1)

        # Balance train
        df_train_balanced = balanceData(df_train)

        # Read valid
        df_valid = pd.read_csv('./myData/valid'+str(i)+'.csv').replace('Expansion', 0).replace('Contingency', 1).replace('Comparison', 2).replace('Temporal', 3).drop('Id', 1)


        # Generate X
        X_train = generateX(df_train_balanced, args.vectorSize)
        X_valid = generateX(df_valid, args.vectorSize)
        X_test = generateX(df_test, args.vectorSize)
        Y_train = list(df_train_balanced['Relation'])
        Y_valid = list(df_valid['Relation'])

        # Training
        lin_clf = svm.LinearSVC()
        lin_clf.fit(X_train, Y_train)


        # Validating
        acc = 0
        for ans,predict in zip(Y_valid, lin_clf.predict(X_valid)):
            if ans == predict:
                 acc += 1
        acc /= df_valid.shape[0]
        accs.append(acc)
        print('acc:', acc)

    acc_avg = sum(accs) / len(accs)
    print('acc_avg:', acc_avg)

    # Gernate ans
    '''
    with open('ans.csv', 'w') as f:
        f.write('Id,Relation\n')
        for Id, predict in zip(df_test['Id'], lin_clf.predict(X_test)):
            f.write(str(Id)+","+numToRelation[predict]+'\n')
    print('Done')
    '''

