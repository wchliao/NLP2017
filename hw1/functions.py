from hanziconv import HanziConv
import jieba
import gensim
import operator

def readDictBig(filename):
	d = {}
	lines = [line.rstrip('\n') for line in open(filename)]
	for line in lines:
		tmp = line.split(' ')
		d[tmp[0]] = tmp[2]
	return d
	

def toSimplified(inputFileName, outputFileName):
	output = open(outputFileName, "w")
	f = open(inputFileName, "r")
	content = f.read()
	simplified = HanziConv.toSimplified(content)
	output.write(simplified)
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
        terms = lines[i].split('\t')[1].split(' ')
        content = []
        for term in terms:
            if term == '':
                continue
            if term in term2POS.keys(): #and ('n' in term2POS[term] or 'i' in term2POS[term]  or 'v' in term2POS[term]) :
                content.append(term)
        contents.append(content)
        i+=1
    
    return contents

def parseTestReviewOut(filename, term2POS):
    contents = []
    lines = [line.rstrip('\n') for line in open(filename)]
    i = 0
    while i < len(lines):
        i += 1
        terms = lines[i].split(' ')
        content = []
        for term in terms:
            if term == '':
                continue
            if term in term2POS.keys(): 
                content.append(term)
        contents.append(content)
        i += 1
    return contents


def parseAspectOut(filename, term2POS):
    contents = []
    lines = [line.rstrip('\n') for line in open(filename)]
    i = 0
    while i < len(lines):
        i += 1
        terms = lines[i].split(' ')
        content = []
        for term in terms:
            if term == '':
                continue
            if term in term2POS.keys():
                content.append(term)
        contents.append(content)
        i += 3
    return contents

def train_word2vec_model(sentences):
	#model = gensim.models.Word2Vec(sentences, size=100, window=5, min_count=5, workers=4)
	model = gensim.models.Word2Vec(sentences, min_count=1)
	return model

def findSimTerm(model, string):
    termSim = {}
    for term, vocab_obj in model.wv.vocab.items():
        termSim[term] = model.wv.similarity(string, term)
    sortedTermSim = sorted(termSim.items(), key=operator.itemgetter(1), reverse=True)
    return sortedTermSim

