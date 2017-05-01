# coding=utf-8

import csv
import argparse

mapping = { u'環境': 0, u'服務': 1, u'交通': 2, u'價格': 3, u'餐廳': 4 }
inv_mapping = [u'環境', u'服務', u'交通', u'價格', u'餐廳']

aspect_terms = [
    frozenset([u"環境", u"客房", u"設備", u"空調"]),
    frozenset([u"服務", u"態度", u"人員"]),
    frozenset([u"交通", u"地理"]),
    frozenset([u"價格", u"房價", u"價錢"]),
    frozenset([u"餐廳", u"早餐"]),
]

def question_proc(ques_file, result, threshold, output_file):
    output = open(output_file, 'w')
    output.write("Id,Label\n")
    with open(ques_file, 'r') as file:
        reader = csv.reader(file, delimiter=',')
        next(reader, None)
        
        for line in reader:
            qid, tid, aspect = int(line[0]), int(line[1]), mapping[line[2]]
            output.write("%d,%d\n" % (
                qid, 1 if result[tid][aspect] > threshold else 0))

    output.close()


def LDA_reader(result_file_name, test_file_name, debug_file):
    debug = None
    if debug_file:
        debug = open(debug_file, 'w')

    result_file = open(result_file_name, 'r')
    test_file = open(test_file_name, 'r')

    result = dict()
    
    for res in result_file.readlines():
        prop = [ float(single.split(':')[1]) for single in res[:-1].split(' ')[:5] ]
        id = int(test_file.readline()[:-1]) # Index

        segs = frozenset(test_file.readline()[:-1].split())  # Parsed segments
        for i, terms in enumerate(aspect_terms):
            if not terms.isdisjoint(segs):  # Got words included
                prop[i] = 1.0

        if debug:
            debug.write("%d [%s]\n" % (id, " ".join(["%d:%f" % (i, p) for i, p in enumerate(prop)])))

        result[id] = prop

    if debug:
        debug.close()
    return result

def run(option):
    res = LDA_reader(option.result_file, option.test_file, option.debug_file)
    question_proc(option.question_file, res, opt.threshold, opt.output_file)


def parse_argu():
    parser = argparse.ArgumentParser(description='Process after JGibbLabeledLDA')
    parser.add_argument('-r', '--result-file', required=True,
                        help='Result file from JGibbLabeledLDA program.')
    parser.add_argument('-t', '--test-file', required=True,
                        help='The parsed test file (not the one from TA).')
    parser.add_argument('-q', '--question-file', required=True,
                        help='The question file from TA.')
    parser.add_argument('-o', '--output-file', required=True,
                        help='Indicate the output .csv file.')
    parser.add_argument('-x', '--threshold', type=float, default=0.2,
                        help='The threshold of probability.')

    parser.add_argument('-d', '--debug-file', help='Debug file output.')

    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_argu()
    run(opt)
