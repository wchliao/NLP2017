#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.merge import Concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D 
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
import pickle

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch_size')
    parser.add_argument('-l', '--learning_rate')
    parser.add_argument('-e', '--epochs')
    parser.add_argument('-fn', '--filter_num')
    parser.add_argument('-fs', '--filter_size')
    parser.add_argument('-cn', '--conv_layer_num')
    parser.add_argument('-d1', '--dropout_prob1')
    parser.add_argument('-d2', '--dropout_prob2')
    parser.add_argument('-op', '--op')
    parser.add_argument('-sh', '--shuffle')
    parser.add_argument('-ad', '--addOneMoreDense')
    parser.add_argument('-dd', '--one_more_dense_dim')
    parser.add_argument('-gl', '--max_over_time_pooling')
    return parser.parse_args()

# python3 cnn_pretrained_cv.py -b 128 -l 0.001 -e 100 -fn 24 -fs 3 -cn 2 -d1 0.25 -d2 0.5 -op test -sh T -ad F -dd 0 -gl T
args = parse_args()

batch_size = int(args.batch_size)#128
learning_rate = float(args.learning_rate)#0.001
epochs =int(args.epochs)#150

filter_num = int(args.filter_num)#32
filter_size = list(map(int, args.filter_size.split(','))) #3,4,5
if len(filter_size) > 1:
    multipleFilterWindows = True
else:
    multipleFilterWindows = False
conv_layer_num  =int(args.conv_layer_num)#2
dropout_prob1 = float(args.dropout_prob1)#0.25
dropout_prob2 = float(args.dropout_prob2)#0.5
op=args.op
if args.shuffle == 'T':
    shuffle = True
else:
    shuffle = False
if args.addOneMoreDense== 'T':
    addOneMoreDense = True
else:
    addOneMoreDense = False
one_more_dense_dim = int(args.one_more_dense_dim)#16
if args.max_over_time_pooling == 'T':
    max_over_time_pooling = True
else:
    max_over_time_pooling = False


# fix random seed for reproducibility
np.random.seed(7)

id2label = dict()
label2id = dict()

# --------- Load word embedding --------- #
words, embeddings = pickle.load(open('/tmp2/eee/polyglot-zh.pkl', 'rb'), encoding='latin1')
print ('%d Zh word embeddings are loaded.' % len(words))
word2id = { w:i for (i,w) in enumerate(words) }

def toVec(seg):
    ans = []
    for word in seg.split():
        index = word2id.get(word)
        if index == None:
            ans.append(0) # <UNK>
        else:
            ans.append(index)
    
    return ans

startS = word2id['<S>']
endS = word2id['</S>']
pad = word2id['<PAD>']

max_review_length = 20
def padding(segs):
    segs = segs[:max_review_length]
    return segs + [ pad ] * ( max_review_length - len(segs) )

def one_hot(i):
    a = np.zeros(4)
    a[i] = 1
    return a

def reverse_one_hot(l):
    for i in range(len(l)):
        if l[i] == 1:
            return i
    return -1

def balanceData(df, op):
    print(op)
    print('Before:')
    counts = df.Relation.value_counts()
    for i in range(4):
        print(i, counts[i]/counts.sum())
    df_expa = df[df['Relation'] == 'Expansion']
    df_cont = df[df['Relation'] == 'Contingency']
    df_comp = df[df['Relation'] == 'Comparison']
    df_temp = df[df['Relation'] == 'Temporal']

    if op == 'sampleExpansion':
        frames = [df_expa.sample(int(df_expa.shape[0] / 2)).copy()]+[df_cont.copy()]+[df_comp.copy()]+[df_temp.copy()]
    elif op == 'train':
        frames = [df_expa.copy()]+[df_cont.copy()]*3+[df_comp.copy()]*5+[df_temp.copy()]*6
    elif op == 'test':
        frames = [df_expa.copy()]+[df_cont.copy()]*3+[df_comp.copy()]*2+[df_temp.copy()]*2
    result = pd.concat(frames, ignore_index='true')
    print('After:')
    counts = result.Relation.value_counts()
    for i in range(4):
        print(i, counts[i]/counts.sum())
    return result

def readTrainORValid(path, op):
    X = []
    y = []
    df = pd.read_csv(path)
    if op != 'None':
        df = balanceData(df, op)
    for i in range(df.shape[0]):
        _, seg1, seg2, ans = df.iloc[i].Id, df.iloc[i].Clause1, df.iloc[i].Clause2, df.iloc[i].Relation
        vec = [ startS ] + toVec(seg1) + [ endS, startS ] + toVec(seg2) + [ endS ]
        X.append( padding(vec) )

        if len(label2id) < 4:
            label2id.setdefault(ans, one_hot(len(label2id)))
            # label2id.setdefault(ans, len(label2id))
        y.append(label2id[ans])
    y = np.asarray(y)
    return X, y


def readTest(path):
    X_test = [] 
    with open(path, 'r') as file:
        file.readline()
        for line in file.readlines():
            _, seg1, seg2 = line[:-1].split(',')
            vec = [ startS ] + toVec(seg1) + [ endS, startS ] + toVec(seg2) + [ endS ]
            X_test.append(padding(vec))
    return X_test

X_train, y_train = readTrainORValid('./data/train.simp.seg', op)
X_test = readTest('./data/test.simp.seg')

# Reverses all label2id.
for label, rep in label2id.items():
    id2label[reverse_one_hot(rep)] = label 
    # id2label[rep] = label 

if shuffle:
    shuffle_indices = np.random.permutation(np.arange(len(y_train)))
    y_train = y_train[shuffle_indices]
    X_train = np.asarray(X_train)
    X_train = list(X_train[shuffle_indices])
    for i in range(len(X_train)):
        X_train[i] = list(X_train[i])

# print ('X_train:', X_train)
# print ('y_train:', y_train)

# truncate and pad input sequences
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# create the model

if multipleFilterWindows:
    branches = []
    for ks in filter_size:
        branch = Sequential()
        branch.add(Embedding(len(embeddings),
                            len(embeddings[0]), # Embedding length
                            weights=[ embeddings ],
                            input_length=max_review_length,
                            trainable=False))
        branch.add(Conv1D(filters=filter_num, kernel_size=ks, padding='valid', activation='relu', strides=1))
        branch.add(GlobalMaxPooling1D())
        branches.append(branch)

    conc = Concatenate()([b.output for b in branches])
    if addOneMoreDense:
        out = (Dense(one_more_dense_dim, activation='relu'))(conc)
        out = Dropout(dropout_prob2)(out)
    else:
        out = Dropout(dropout_prob2)(conc)
    out = Dense(4, activation='softmax')(out)

    model = Model([b.input for b in branches], out)


    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    X_train = np.asarray(X_train)
    model.fit([X_train]*len(filter_size), y_train, epochs=epochs, batch_size=batch_size)

else:
    model = Sequential()
    model.add(Embedding(
        len(embeddings),
        len(embeddings[0]), # Embedding length
        weights=[ embeddings ],
        input_length=max_review_length,
        trainable=False
    ))
    if max_over_time_pooling:
        for i in range(conv_layer_num):
            model.add(Conv1D(filters=filter_num, kernel_size=filter_size, padding='same', activation='relu'))
        model.add(GlobalMaxPooling1D())
        model.add(Dropout(dropout_prob1))
    else: 
        if conv_layer_num == 2:
            for i in range(2):
                model.add(Conv1D(filters=filter_num, kernel_size=filter_size, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout_prob1))
        elif conv_layer_num == 3:
            for i in range(2):
                model.add(Conv1D(filters=filter_num, kernel_size=filter_size, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout_prob1))
            model.add(Conv1D(filters=filter_num, kernel_size=filter_size, padding='same', activation='relu'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(dropout_prob1))
        model.add(Flatten())


    if addOneMoreDense:
        model.add(Dense(one_more_dense_dim, activation='relu'))
    model.add(Dropout(dropout_prob2))
    model.add(Dense(4, activation='softmax'))


    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
    print(model.summary())
    model.fit([X_train, X_train, X_train], y_train, epochs=epochs, batch_size=batch_size)


X_test= np.asarray(X_test)
predict = model.predict([X_test, X_test, X_test], batch_size=batch_size)
print ('Predict:', predict)

with open('./log/pretrain'+'_'+str(filter_num)+'_'+str(args.filter_size)+'_'+str(conv_layer_num)+'_'+str(dropout_prob1)+'_'+str(dropout_prob2)+'_'+str(addOneMoreDense)+'_'+str(op)+'_'+str(shuffle)+'_'+str(batch_size)+'_'+str(learning_rate)+'_'+str(epochs)+'.csv', 'w') as file:
    file.write('Id,Relation\n')
    for id, ans in enumerate(predict):
        file.write('%d,%s\n' % (id + 6639, id2label[np.argmax(ans)]))

with open('result.csv', 'w') as file:
    file.write('Id,Relation\n')
    for id, ans in enumerate(predict):
        file.write('%d,%s\n' % (id + 6639, id2label[np.argmax(ans)]))
print ('All labels predicted!')
