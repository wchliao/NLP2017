# coding=utf-8

class Reader:
    # trim = u''':!),.:;?]}¢'"、。〉》」』】〕〗〞
            # ︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂
            # ﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵
            # ︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…'''
    trim = u",.;!，。？！"
    @staticmethod
    def get_sentences(segs):
        sentences = []
        sent = []
        for word in segs:
            if word != '' and word not in Reader.trim:
                sent.append(word)
            elif len(sent) > 0:  # Pushes new sentence.
                sentences.append(sent[:])
                sent = []
        sentences.append(sent[:])

        return sentences


    @staticmethod
    def aspect(filename):
        res = []
        with open(filename, 'r') as file:
            line = file.readline()
            while line:
                num = int(line[:-1])
                # Parse sentences list
                line = file.readline()
                segs = line[:-1].split(' ')
                sentences = Reader.get_sentences(segs)

                line = file.readline()  # Positive line 
                pos = line.split()

                line = file.readline()  # Negative line
                neg = line.split()
                
                res.append((num, sentences, pos, neg))
                line = file.readline()

        return res

    @staticmethod
    def polarity(filename):
        res = []
        with open(filename, 'r') as file:
            line = file.readline()
            while line:
                segs = line[:-1].split('\t')
                val = int(segs[0])
                sentences = Reader.get_sentences(segs[1].split(' '))
                res.append((val, sentences))
                line = file.readline()

        return res

    @staticmethod
    def test(filename):
        res = []
        with open(filename, 'r') as file:
            line = file.readline()
            while line:
                num = int(line[:-1])
                line = file.readline()
                sentences = Reader.get_sentences(line[:-1].split(' '))
                res.append((num, sentences))
                line = file.readline()

        return res
