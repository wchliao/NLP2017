#encoding=utf-8
from __future__ import print_function

# Natural language processing module
import jieba
import jieba.analyse
from textrank4zh import TextRank4Keyword, TextRank4Sentence

class Segment:
    def __init__(self, new_file="word.new",
                 stopword_file="external/stopwords_tw.txt",
                 dict=None, user_dict=None):
        print ("Segment-er initialized.")
        if dict:
            jieba.set_dictionary(dict)
        if user_dict:
            jieba.load_userdict(user_dict)
        
        self.new_file = open(new_file, "a")
        self.tr = TextRank4Keyword(stop_words_file=stopword_file)

    def add_words(self, words):
        """
        Add words to jieba module and files for record.
        TODO: Add new word to corpus?

        Args:
            words ([(string, int)]): list of (term, score)
        """
        for word, val in words:
            jieba.add_word(word)
            self.new_file.write(word + "\n")


    def parse(self, content):
        """
        Parse paragraph into word segments.
        
        Args:
            content (string): paragraph to be parsed.
        """
        return list(jieba.cut(self.trim_punct(content), cut_all=False))

    def get_keywords(self, content, KEY_NUM=10):
        """
        Get keywords for the given paragraph.

        Args:
            content (string): paragraph to be processed.
            KEY_NUM (int): Maximum number of keywords 
                           retrieved by each of the methods.
        """
        keys = set()

        self.tr.analyze(text=content, lower=True, window=5)
        words = [ x.word for x in self.tr.get_keywords(KEY_NUM, word_min_len=2) ]
        keys = keys | set(words)

        words = self.tr.get_keyphrases(keywords_num=KEY_NUM, min_occur_num=2)
        keys = keys | set(words)

        words = jieba.analyse.extract_tags(content, topK=KEY_NUM)
        keys = keys | set(words)

        words = jieba.analyse.textrank(content, topK=KEY_NUM, withWeight=False)
        keys = keys | set(words)

        return list(keys)

    @staticmethod
    def trim_punct(post):
        table = { ord(c) : " " for c in u''':!),.:;?]}¢'"、。〉》」』】〕〗〞
                 ︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂
                 ﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵
                 ︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…''' }
        return post.translate(table)
