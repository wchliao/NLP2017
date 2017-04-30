# coding=utf-8

import argparse
from Reader import Reader

def main(opt):
    mapping = { u'環境': "0", u'服務': "1", u'交通': "2", u'價格': "3", u'餐廳': "4" }
    print ("The mapping: %s" % (", ".join("%s: %s" % pair for pair in mapping.items())))

    file = open(opt.output, 'w')

    if opt.aspect:  # Labeled
        labeled = Reader.aspect(opt.aspect)
        for id, sentences, good, bad in labeled:
            file.write("[%s] %s\n" % (
                " ".join(map(lambda x: mapping[x], good + bad)),
                " ".join([ x for sent in sentences for x in sent ])
            ))

    if opt.polarity:  # Unlabeled
        unlabeled = Reader.polarity(opt.polarity)
        for val, sentences in unlabeled:
            file.write(" %s\n" % " ".join(
                [ x for sent in sentences for x in sent ]))

    if opt.test:  # Unlabeled
        unlabeled = Reader.test(opt.test)
        for val, sentences in unlabeled:
            file.write(" %s\n" % " ".join(
                [ x for sent in sentences for x in sent ]))

    file.close()

def parse_args():
    parser = argparse.ArgumentParser(description='Collocation processing.')

    # Input files indication.
    parser.add_argument('-a', '--aspect', help='Indicate aspect_review input.')
    parser.add_argument('-p', '--polarity', help='Indicate polarity_review input.')
    parser.add_argument('-t', '--test', help='Indicate test_review input.')

    parser.add_argument('-o', '--output', required=True, 
                        help='Indicate output file.')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()
    main(opt)
