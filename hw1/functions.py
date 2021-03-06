# -*- coding: utf-8 -*-

from hanziconv import HanziConv
import jieba
import gensim
import operator
import re

def readDictBig(filename):
	d = {}
	lines = [line.rstrip('\n') for line in open(filename)]
	for line in lines:
		tmp = line.split(' ')
		d[tmp[0]] = tmp[2]
	return d
	
def split2sentences(comment):
    sentences = re.split('[,.～！，。；]', comment)
    return sentences

def comment2sentence(term2POS, comment, contents, isPolarity):
    sentences = split2sentences(comment)
    for i, sentence in enumerate(sentences):
        if isPolarity and i == 0:
            terms = sentence.split('\t')[1].split(' ')
        else:
            terms = sentence.split(' ')
        content = []
        for term in terms:
            if term == '':
                continue
            if term in term2POS.keys():
                content.append(term)
        contents.append(content)
    return contents

def toSimplified(inputFileName, outputFileName):
	output = open(outputFileName, "w")
	f = open(inputFileName, "r")
	content = f.read()
	simplified = HanziConv.toSimplified(content)
	output.write(simplified)
	f.close()
	output.close()

def toTraditional(inputFileName, outputFileName):
	output = open(outputFileName, "w")
	f = open(inputFileName, "r")
	content = f.read()
	trad = HanziConv.toTraditional(content)
	output.write(trad)
	f.close()
	output.close()

def segment(input_str):
	seg_list = jieba.cut(input_str, cut_all=False)
	print(', '.join(seg_list));

def parsePolarityOut(filename, term2POS):
    contents = []
    lines = [line.rstrip('\n') for line in open(filename)]
    i = 0
    while i < len(lines):
        contents = comment2sentence(term2POS, lines[i], contents, True)
        i+=1
    
    return contents

def parseTestReviewOut(filename, term2POS):
    contents = []
    lines = [line.rstrip('\n') for line in open(filename)]
    i = 0
    while i < len(lines):
        i += 1
        contents = comment2sentence(term2POS, lines[i], contents, False)
        i += 1
    return contents

def parseAspectOut(filename, term2POS):
    contents = []
    lines = [line.rstrip('\n') for line in open(filename)]
    i = 0
    while i < len(lines):
        i += 1
        contents = comment2sentence(term2POS, lines[i], contents, False)
        i += 3
    return contents

def train_word2vec_model(sentences):
	model = gensim.models.Word2Vec(sentences, size=150, window=10, min_count=30, workers=4)
	return model

def findSimTerm(model, string):
    termSim = {}
    for term, vocab_obj in model.wv.vocab.items():
        termSim[term] = model.wv.similarity(string, term)
    sortedTermSim = sorted(termSim.items(), key=operator.itemgetter(1), reverse=True)
    return sortedTermSim

def getSimTerms(model, baseTerms, threshold):
    newTerms = set()
    for i, term in enumerate(baseTerms):
        newTerms.add(term)
        sortedTermSim = findSimTerm(model, term)
        for newTerm, sim in sortedTermSim:
            if sim > threshold[i]:
                newTerms.add(newTerm)
    return newTerms

def readTestReview(filename):
    testReviewDict = {}
    lines = [line.rstrip('\n') for line in open(filename)]
    i = 0
    while i < len(lines):
        testReviewDict[int(lines[i])] = lines[i+1]
        i+=2
        
    return testReviewDict



def containTerms(terms, aspectTerms):
    #terms = review.split(' ')
    for term in terms:
        if term in aspectTerms:
            return True
        
def decideMention(review, aspect, serviceNewTerms, envNewTerms, priceNewTerms, trafficNewTerms, restaurantNewTerms):
    if aspect == '服務':
        if containTerms(review, serviceNewTerms):
            return 2
    elif aspect == '環境':
        if containTerms(review, envNewTerms):
            return 2
    elif aspect == '價格':
        if containTerms(review, priceNewTerms):
            return 2
    elif aspect == '交通':
        if containTerms(review, trafficNewTerms):
            return 2
    elif aspect == '餐廳':
        if containTerms(review, restaurantNewTerms):
            return 2
    return 0
   

def decideAspect(review, serviceNewTerms, envNewTerms, priceNewTerms, trafficNewTerms, restaurantNewTerms):
    if containTerms(review, serviceNewTerms):
        return 1
    if containTerms(review, envNewTerms):
        return 2
    if containTerms(review, priceNewTerms):
        return 3
    if containTerms(review, trafficNewTerms):
        return 4
    if containTerms(review, restaurantNewTerms):
        return 5
    return 0
