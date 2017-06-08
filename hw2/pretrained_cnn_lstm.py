#!/usr/bin/env python3

# LSTM and CNN for sequence classification in the IMDB dataset
import numpy as np
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
import pickle

# fix random seed for reproducibility
# np.random.seed(7)

# load the dataset but only keep the top n words, zero the rest
top_words = 5000
# (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=top_words)

X_train = []; y_train = []
X_test = []
id2label = dict()
label2id = dict()

# --------- Load word embedding --------- #
words, embeddings = pickle.load(open('./polyglot-zh.pkl', 'rb'), encoding='latin1')
print ('%d Zh word embeddings are loaded.' % len(words))
word2id = { w:i for (i,w) in enumerate(words) }

print (words[:10])

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

with open('./data/train.simp.seg', 'r') as file:
    file.readline()
    for line in file.readlines():
        id, seg1, seg2, ans = line[:-1].split(',')
        vec = [ startS ] + toVec(seg1) + [ endS, startS ] + toVec(seg2) + [ endS ]
        X_train.append( padding(vec) )

        if len(label2id) < 4:
            label2id.setdefault(ans, one_hot(len(label2id)))
            # label2id.setdefault(ans, len(label2id))
        y_train.append(label2id[ans])

with open('./data/test.simp.seg', 'r') as file:
    file.readline()
    for line in file.readlines():
        id, seg1, seg2 = line[:-1].split(',')
        vec = [ startS ] + toVec(seg1) + [ endS, startS ] + toVec(seg2) + [ endS ]
        X_test.append(padding(vec))


# Reverses all label2id.
for label, rep in label2id.items():
    id2label[reverse_one_hot(rep)] = label 
    # id2label[rep] = label 

# print ('X_train:', X_train)
# print ('y_train:', y_train)

# truncate and pad input sequences
# X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
# X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)

y_train = np.asarray(y_train)

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
model.add(LSTM(50))
# model.add(MaxPooling1D(pool_size=2))
# model.add(Dense(1, activation='sigmoid'))
model.add(Dense(4, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, epochs=60, batch_size=64)

# Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))

predict = model.predict(X_test, batch_size=64)
print ('Predict:', predict)
with open('result_lstm.csv', 'w') as file:
    file.write('Id,Relation\n')
    for id, ans in enumerate(predict):
        file.write('%d,%s\n' % (id + 6639, id2label[np.argmax(ans)]))

print ('All labels predicted!')
