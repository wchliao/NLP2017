#encoding=utf-8
from __future__ import print_function

import re
import os
import sys
import jieba

from taggerlib.segment import Segment

from nltk.tag import StanfordNERTagger


"""
Example usage:
    from taggerlib.term_extract import Extraction

    ext = Extraction(6)
    term_count, segments, ners = ext.run(posts)
"""

class Extraction:
    """
    Term extraction. Extract new terms from multiple articles by calculating
    Symmetric Conditional Probability and Context Dependency (SCPCD) score.

    Attributes:
        counter (dict(): string => [int, [int, set()], [int, set()]]): Mapping
            term to [0] frequency count, 
            [1] [front empty count, set(front character)] and
            [2] [back empty count, set(back character)]
        maxgram (int): maximum length of terms extracted.
    """

    def __init__(self, n_gram=8): 
        """
        Initialize extraction machine.

        Args:
            n_gram (int): maximum length of terms extracted. Default is 8.
        """
        self.counter = dict()
        self.maxgram = n_gram
        self.segmenter = Segment(dict="external/dict.txt.big") 
        self.nertagger = StanfordNERTagger(
            model_filename="external/stanford/chinese.misc.distsim.crf.ser.gz",
            path_to_jar="external/stanford/stanford-ner.jar"
        )
        print ("Extraction class initialized.")

    def run(self, posts, threshold=1.5):
        """
        Main process of Extraction, running the preprocessing and SCPCD score
        calculation.

        Args:
            posts ([string]): the collection of articles.

        Returns:
            (
                dict(): mapping from term to its SCPCD score.
                list(list()): For each of the post, returns the segments of 
                              each post.
                dict(): mapping NER tag to set of terms.
            )
        """

        print ("Extraction process starts.")
        
        for post in posts:
            self.parse_post(post)
        print ("Done parsing %d posts." % len(posts))
        # print ("首：", self.counter['首'])
        # print ("都：", self.counter['都'])

        ans = dict()
        for term in self.counter.keys():
            if len(term) > 1:
                ans[term] = self.SCP_CD(term)

        ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)
        self.segmenter.add_words(filter(lambda x: x[1] > threshold, ans))
        post_segment = []
        ners = dict()
        
        for post in posts:
            print ("Post: %s" % post[:35])
            segments = self.segmenter.parse(post)
            post_segment.append(segments)
            for term, ner in self.nertagger.tag(segments):
                if ner != "O":
                    sys.stdout.write("[ %s (%s) ]" % (term, ner))
                    ners.setdefault(ner, dict())
                    ners[ner].setdefault(term, 0)
                    ners[ner][term] += 1
            print ("\n----------------------------\n")
            sys.stdout.flush()
        
        return ans, post_segment, ners

    def opt_parse(self, posts, score_threshold=1.5):
        score = self.new_term(posts)
        count = 0
        for term, s in score:
            if s >= score_threshold:
                count += 1
                self.segmenter.add_words(ans[:threshold])
            else:
                break
        print ("Opt_parse (threshold: %f) added %d words." % (
            score_threshold, count))

        segments = []
        for post in posts:
            segments.append(self.segmenter.parse(post))
        return segments

    def new_term(self, posts):
        for post in posts:
            self.parse_post(post)
        print ("Done parsing %d posts." % len(posts))

        ans = dict()
        for term in self.counter.keys():
            if len(term) > 1:
                ans[term] = self.SCP_CD(term)

        ans = sorted(ans.items(), key=lambda x: x[1], reverse=True)
        return ans


    def SCP_CD(self, term):
        """
        Combination of Symmetric Conditional Probability and Context Dependency,
        The higher SCPCD score a term get, the higher probability of being a 
        independent term.

        Args:
            term (string): the term, whose length > 1, that is to be calculated
                           SCPCD score.

        Returns:
            int: SCPCD score
        """

        count, left_counter, right_counter = self.counter[term]

        left = left_counter[0] + len(left_counter[1])
        right = right_counter[0] + len(right_counter[1])

        sumpart = 0
        for br in range(1, len(term)):
            # print ("front: %s, back: %s" % (term[:br], term[br:]))
            sumpart += self.counter[ term[:br] ][0] * self.counter[ term[br:] ][0]
        if sumpart == 0:
            print ("%s sumpart = 0" % term)

        return (len(term) - 1) * left * right / sumpart * count

    def parse_post(self, post):
        """
        Parse single article into n-grams for SCP_CD.

        Args:
            n_gram (int): maximum length of term.
            post (string): article content.

        Returns:
            None
        """
        sentences = re.findall(r"[\w']+", post)
        for sentence in sentences:
            length = len(sentence)
            for n in range(1, self.maxgram + 1):
                for start in range(length - n):
                    term = sentence[start : start + n]

                    count, front, back = self.counter.setdefault(
                        term, [0, [0, set()], [0, set()]]
                    )
                    self.counter[term][0] += 1

                    if start == 0:
                        front[0] += 1
                        back[1].add(sentence[start + n])
                    elif start >= length - n:
                        front[1].add(sentence[start - 1])
                        back[0] += 1
                    else:
                        front[1].add(sentence[start - 1])
                        back[1].add(sentence[start + n])

                        





