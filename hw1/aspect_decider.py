# -*- coding: utf-8 -*-

import argparse
import sys
import functions as func
import operator
import pandas as pd
import numpy as np
import re

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-p', '--polarity_simpl_file', help='give polarity simpl file here.')
    parser.add_argument('-a', '--aspect_simpl_file', help='give aspect simpl file here.')
    parser.add_argument('-t', '--test_simpl_file', help='give test review simpl file here.')
    parser.add_argument('-q', '--question_file', help='give question csv file here.')
    parser.add_argument('-o', '--output_file', help='give output file here.')

    parser.add_argument('-d', '--threshold', help='similarity threshold')

    return parser.parse_args()

def parseData(args):
    polarity_contents = func.parsePolarityOut(args.polarity_simpl_file, term2POS)    #'data/polarity_simpl.out', term2POS)
    aspect_contents = func.parseAspectOut(args.aspect_simpl_file, term2POS)     #'data/aspect_simpl.out', term2POS)
    testReview_contents = func.parseTestReviewOut(args.test_simpl_file, term2POS)#'data/test_simpl.out', term2POS)

    contents = polarity_contents+aspect_contents+testReview_contents
    return contents

# Train Word2Vec model
def trainWord2Vec(contents):
    model = func.train_word2vec_model(contents)
    return model

# Find similar terms
def findSimilarTermsOfAllAspects(model, serviceTerms, envTerms, priceTerms, trafficTerms, restaurantTerms, threshold):
    serviceNewTerms = func.getSimTerms(model, serviceTerms, [threshold, threshold, threshold])
    envNewTerms = func.getSimTerms(model, envTerms, [threshold, threshold, threshold, threshold])
    priceNewTerms = func.getSimTerms(model, priceTerms, [threshold, threshold])
    trafficNewTerms = func.getSimTerms(model, trafficTerms, [threshold, threshold])
    restaurantNewTerms = func.getSimTerms(model, restaurantTerms, [threshold, threshold])
    return serviceNewTerms, envNewTerms, priceNewTerms, trafficNewTerms, restaurantNewTerms


# DecideAspect for each sentence and write to file
def decideAspectToFile(testReviewDict, serviceNewTerms, envNewTerms, priceNewTerms, trafficNewTerms, restaurantNewTerms, filename):
    #f = open('Aspect_per_sentence.csv', 'w')
    f = open(filename, 'w')
    for Id, comment in testReviewDict.items():
        sentences = func.split2sentences(comment)
        for sentence in sentences:
            f.write(str(Id)+","+str(func.decideAspect(sentence, serviceNewTerms, envNewTerms, priceNewTerms, trafficNewTerms, restaurantNewTerms))+'\n')
    f.close()



# Count label (0 or not 0) according to input file
def countLabelTofile(testReviewDict, testDF, serviceNewTerms, envNewTerms, priceNewTerms, trafficNewTerms, restaurantNewTerms):
    f = open('result.csv', 'w')
    f.write('Id,Label\n')
    for i in range(testDF.shape[0]):
        Id, Review_id, Aspect = testDF.iloc[i]
        f.write(str(Id)+","+str(func.countLabel(testReviewDict[int(Review_id)], str(Aspect), serviceNewTerms, envNewTerms, priceNewTerms, trafficNewTerms, restaurantNewTerms))+'\n')
    f.close()

if __name__ == '__main__':
    
    args = parse_args()

    # Term to POS tag
    term2POS = func.readDictBig('external/dict.txt.big')
    contents = parseData(args)
    model = trainWord2Vec(contents)

    # Head aspect terms
    serviceTerms = ['服务','态度','人员']
    envTerms = ['环境','客房','设备','空调']
    priceTerms = ['价格','房价']
    trafficTerms = ['交通','地理']
    restaurantTerms = ['餐厅','早餐']


    # Read input file for submission
    testReviewDict = func.readTestReview(args.test_simpl_file)    #'data/test_simpl.out')
    #testDF = pd.read_csv(args.question_file)                     #'data/test.csv')

    serviceNewTerms, envNewTerms, priceNewTerms, trafficNewTerms, restaurantNewTerms = findSimilarTermsOfAllAspects(model, serviceTerms, envTerms, priceTerms, trafficTerms, restaurantTerms, float(args.threshold))

    print(len(serviceNewTerms))
    print(len(envNewTerms))
    print(len(priceNewTerms))
    print(len(trafficNewTerms))
    print(len(restaurantNewTerms))

    #print(serviceNewTerms)
    #print(envNewTerms)
    #print(priceNewTerms)
    #print(trafficNewTerms)
    #print(restaurantNewTerms)

    decideAspectToFile(testReviewDict, serviceNewTerms, envNewTerms, priceNewTerms, trafficNewTerms, restaurantNewTerms, args.output_file)

    print('End')
