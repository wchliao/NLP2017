import tensorflow as tf
import numpy as np
import pandas as pd
import os
import time
import datetime
from func.text_cnn import TextCNN
import func.data_helpers as data_helpers
from tensorflow.contrib import learn
from collections import Counter
import argparse

# python3 cnn_github.py -ed 128 -fs 3,4,5 -fn 128 -d 0.8 -l 0 -r 0.0001 -b 128 -ep 200
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-ed', '--embedding_dim', help='')
    parser.add_argument('-fs', '--filter_sizes', help='')
    parser.add_argument('-fn', '--num_filters', help='')
    parser.add_argument('-d', '--dropout_keep_prob', help='')
    parser.add_argument('-l', '--l2_reg_lambda', help='')
    parser.add_argument('-r', '--learning_rate', help='')
    parser.add_argument('-b', '--batch_size', help='')
    parser.add_argument('-ep', '--num_epochs', help='')

    return parser.parse_args()


def labelToOneHot(array):
    dim = np.max(array) + 1
    oneHot = np.zeros(shape=(len(array), dim))
    for i,label in enumerate(array):
        oneHot[i][label] = 1
    return oneHot

def loadData(filename):
    df = pd.read_csv(filename).replace('Expansion', 0).replace('Contingency', 1).replace('Comparison', 2).replace('Temporal', 3).drop('Id', 1)
    x_text = list(df.Clause1 + " " + df.Clause2)
    y = labelToOneHot(np.asarray(df.Relation))
    return x_text, y

def train_step(sess, cnn, x_batch, y_batch, train_op):
    """
    A single training step
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
    }
    _, loss, accuracy = sess.run(
        [train_op, cnn.loss, cnn.accuracy],
        feed_dict)

    return loss

def dev_step(sess, cnn, x_batch, y_batch):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      cnn.input_x: x_batch,
      cnn.input_y: y_batch,
      cnn.dropout_keep_prob: 1.0
    }

    accuracy = sess.run(cnn.accuracy, feed_dict)
    return accuracy

def generate_ans(sess, cnn, x_test):
    """
    Evaluates model on a dev set
    """
    feed_dict = {
      cnn.input_x: x_test,
      cnn.dropout_keep_prob: 1.0
    }
    ans = sess.run([cnn.predictions],feed_dict)
    return ans

def writeAns(sess, cnn, x_test, numToRelation, args):
    ans = generate_ans(sess, cnn, x_test)

    with open('./log/ans_cnn'+str(args.embedding_dim)+'_'+str(args.filter_sizes)+'_'+str(args.dropout_keep_prob)+'_'+str(args.l2_reg_lambda)+'_'+str(args.dropout_keep_prob)+'_'+str(args.batch_size)+'_'+str(args.num_epochs)+'.csv', 'w') as f:
        f.write('Id,Relation\n')
        for i, Id in enumerate(df_test['Id']):
            f.write(str(Id)+","+numToRelation[ans[0][i]]+'\n')

    print('=====================================================')
    for k,v in Counter(ans[0]).items():
        print(numToRelation[k], ":", v*100/sum(Counter(ans[0]).values()),'%')
    print('=====================================================')
    print('Done')


def trainAndValid(vocab_processor, i):
    x_train_text, y_train = loadData('./myData/train'+str(i)+'.csv')              #Not balanced
    x_train = np.array(list(vocab_processor.fit_transform(x_train_text)))
    x_dev_text, y_dev = loadData('./myData/valid'+str(i)+'.csv')
    x_dev = np.array(list(vocab_processor.fit_transform(x_dev_text)))

    # shuffle?

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(float(args.learning_rate))
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Initialize all variables
            sess.run(tf.global_variables_initializer())


            # Generate batches
            batches = data_helpers.batch_iter(list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            num_batches_per_epoch = int((len(x_train)-1)/FLAGS.batch_size) + 1

            # Training loop. For each batch...
            epoch_i = 0
            loss_list = []
            final_acc_eval = 0
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                loss_ret = train_step(sess, cnn, x_batch, y_batch, train_op)
                loss_list.append(loss_ret)
                current_step = tf.train.global_step(sess, global_step)
                if (current_step+1) % num_batches_per_epoch == 0:
                    epoch_i += 1
                    print("Train:\t", epoch_i, '\tloss =', sum(loss_list)/len(loss_list))
                    loss_list = []
                    acc_eval = dev_step(sess, cnn, x_dev, y_dev)
                    print("Evalu:\t", epoch_i, '\tacc =', acc_eval)
                    if epoch_i == FLAGS.num_epochs:
                        final_acc_eval = acc_eval

            #writeAns(sess, cnn, x_test, numToRelation, args)
            return final_acc_eval

if __name__ == '__main__':

    args = parse_args()
    np.random.seed(10)

    # Parameters
    # ==================================================

    # Data loading params
    tf.flags.DEFINE_float("dev_sample_percentage", .2, "Percentage of the training data to use for validation")

    # Model Hyperparameters
    tf.flags.DEFINE_integer("embedding_dim", int(args.embedding_dim), "Dimensionality of character embedding (default: 128)")
    tf.flags.DEFINE_string("filter_sizes", str(args.filter_sizes), "Comma-separated filter sizes (default: '3,4,5')")
    tf.flags.DEFINE_integer("num_filters", int(args.num_filters), "Number of filters per filter size (default: 128)")
    tf.flags.DEFINE_float("dropout_keep_prob", float(args.dropout_keep_prob), "Dropout keep probability (default: 0.5)")
    tf.flags.DEFINE_float("l2_reg_lambda", float(args.l2_reg_lambda), "L2 regularization lambda (default: 0.0)")

    # Training parameters
    tf.flags.DEFINE_integer("batch_size", int(args.batch_size), "Batch Size (default: 64)")
    tf.flags.DEFINE_integer("num_epochs", int(args.num_epochs), "Number of training epochs (default: 200)")

    # Misc Parameters
    tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
    tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")

    FLAGS = tf.flags.FLAGS
    FLAGS._parse_flags()
    print("\nParameters:")
    for attr, value in sorted(FLAGS.__flags.items()):
        print("{}={}".format(attr.upper(), value))
    print("")

    numToRelation = ['Expansion', 'Contingency', 'Comparison', 'Temporal']

    # Data Preparation
    # ==================================================

    # Load data
    x_text, y = loadData('./data/train.simp.seg') #Not balanced
    df_test = pd.read_csv('./data/test.simp.seg')
    x_test_text = list(df_test.Clause1 + " " + df_test.Clause2)


    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text]+[len(x.split(" ")) for x in x_test_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    x_test = np.array(list(vocab_processor.fit_transform(x_test_text)))

    # Randomly shuffle data
    #shuffle_indices = np.random.permutation(np.arange(len(y)))
    #x_shuffled = x[shuffle_indices]
    #y_shuffled = y[shuffle_indices]


    # Training
    # ==================================================

    acc_eval_list = []
    for i in range(5):
        acc_eval_list.append(trainAndValid(vocab_processor, i))

    print('Avg_acc:', sum(acc_eval_list) / len(acc_eval_list))
    
