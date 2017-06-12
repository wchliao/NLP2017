#!/usr/bin/env python3

import numpy as np
import pandas as pd
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.optimizers import Adam
import pickle

batch_size = 128
epochs = 100

# fix random seed for reproducibility
np.random.seed(7)

id2label = dict()
label2id = dict()

# --------- Load word embedding --------- #
words, embeddings = pickle.load(open('/tmp2/eee/polyglot-zh.pkl', 'rb'), encoding='latin1')
print ('%d Zh word embeddings are loaded.' % len(words))
print ('Embedding size', len(embeddings[0]))
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
    if op == 'train':
        frames = [df_expa.copy()]+[df_cont.copy()]*3+[df_comp.copy()]*5+[df_temp.copy()]*6
    elif op == 'valid':
        frames = [df_expa.copy()]+[df_cont.copy()]*3+[df_comp.copy()]*2+[df_temp.copy()]*2
    result = pd.concat(frames, ignore_index='true')
    print('After:')
    counts = result.Relation.value_counts()
    for i in range(4):
        print(i, counts[i]/counts.sum())
    return result

def readTrainORValid(path, op=None):
    X = []
    y = []
    df = pd.read_csv(path)
    if op != None:
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

#X_train, y_train = readTrainORValid('./data/train.simp.seg')
X_train, y_train = readTrainORValid('./myData/train0.csv', 'train')
X_valid, y_valid = readTrainORValid('./myData/valid0.csv', None)
X_test = readTest('./data/test.simp.seg')

# Reverses all label2id.
for label, rep in label2id.items():
    id2label[reverse_one_hot(rep)] = label 
    # id2label[rep] = label 

shuffle_indices = np.random.permutation(np.arange(len(y_train)))
X_train = np.asarray(X_train)
X_train = X_train[shuffle_indices]
y_train = y_train[shuffle_indices]
X_train = list(X_train[shuffle_indices])
for i in range(len(X_train)):
    X_train[i] = list(X_train[i])

# print ('X_train:', X_train)
# print ('y_train:', y_train)

# truncate and pad input sequences
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)


# create the model
model = Sequential()
model.add(Embedding(
    len(embeddings),
    len(embeddings[0]), # Embedding length
    weights=[ embeddings ],
    input_length=max_review_length,
    trainable=False
))

model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Dropout(0.25))

#model.add(Conv1D(filters=32, kernel_size=4, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Dropout(0.25))

#model.add(Conv1D(filters=32, kernel_size=5, padding='same', activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Dropout(0.25))

model.add(Flatten())
#model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))

adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

# Final evaluation of the model
scores = model.evaluate(X_valid, y_valid, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

predict = model.predict(X_test, batch_size=batch_size)
print ('Predict:', predict)
with open('result.csv', 'w') as file:
    file.write('Id,Relation\n')
    for id, ans in enumerate(predict):
        file.write('%d,%s\n' % (id + 6639, id2label[np.argmax(ans)]))

print ('All labels predicted!')
