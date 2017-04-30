from __future__ import print_function
from Reader import Reader
from gensim import corpora, models



class LDA:
    def __init__(self):
        print ('LDA model initialized.')
        print ('Please use `build_model` or `load_model` to initiate model.')

    def build_model(self, aspect_file=None, polarity_file=None, test_file=None,
                 stopword_file='external/stopwords_tw.txt',
                 N_passes=1, N_TOPIC=5,
                 flat_post=False, multicore=False):
        # Reads the files
        self.aspect = Reader.aspect(aspect_file) if aspect_file else []
        print ('Done loading aspect file.')
        self.polarity = Reader.polarity(polarity_file) if polarity_file else []
        print ('Done loading polarity file.')
        self.test = Reader.test(test_file) if test_file else []
        print ('Done loading test file.')

        # Read stopwords set
        stopwords = set()
        with open(stopword_file, 'r') as file:
            for line in file.readlines():
                stopwords.add(line[:-1])

        def toStrList(ll):
            if not flat_post:
                return [ list(filter(lambda word: word not in stopwords, sent))
                        for res in ll for sent in res[1] ]
            # return [ [ 
                # word for sent in res[1] for word in sent 
            # ] for res in ll ]

            return [ list(filter(lambda word: word not in stopwords, [
                word for sent in res[1] for word in sent
            ])) for res in ll ]

        texts = toStrList(self.aspect) 
        texts += toStrList(self.polarity)
        texts += toStrList(self.test)

        self.dict = corpora.Dictionary(texts)
        self.bag = [self.dict.doc2bow(text) for text in texts]
        print ('Done pre-processing bag-of-words for LDA.')

        model_func = models.ldamodel.LdaModel
        if multicore:
            model_func = models.ldamulticore.LdaMulticore

        self.model = model_func(self.bag, 
                                num_topics=N_TOPIC, 
                                id2word=self.dict, 
                                passes=N_passes)

    def dump_result(self, filename):
        with open(filename, 'w') as file:
            for doc in self.aspect:
                file.write(">>> [+] (%s) [-] (%s)\n" % (
                    ",".join(doc[2]), ",".join(doc[3])
                ))
                for sent in doc[1]:
                    res = self.model[self.dict.doc2bow(sent)]
                    file.write("%s :: %s\n" % (
                        " ".join([ "%.3f" % x[1] for x in res ]),
                        " ".join(sent)
                    ))
                file.write("\n")


    def save_model(self, prefix_name):
        self.model.save(prefix_name + '.lda')
        self.dict.save(prefix_name + '.dict')
        print ('Saved two files: %s.lda & %s.dict' % (prefix_name, prefix_name))

    def load_model(self, prefix_name) :
        self.model.load(prefix_name + '.lda')
        self.dict.load(prefix_name + '.dict')
        print ('Loaded two files: %s.lda & %s.dict' % (prefix_name, prefix_name))


def parse_args():
    parser = argparse.ArgumentParser(description='Collocation processing.')

    # Input files indication.
    parser.add_argument('-a', '--aspect', help='Indicate aspect_review input.')
    parser.add_argument('-p', '--polarity', help='Indicate polarity_review input.')
    parser.add_argument('-t', '--test', help='Indicate test_review input.')

    return parser.parse_args(sys.argv[1:])

if __name__ == '__main__':
   opt = parse_args() 
   run(opt)

