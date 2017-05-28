"""
Preprocessing module for National Taiwan University Computer Science dept.
Natural Language Processing course Second Homework.

Target: connect to Paper: A Recurrent Neural Model with Attention for
the Recognition of Chinese Implicit Discourse Relations Samuel (ACL '17)

Notes:
    1. There is no POS TAG used in the original source code.
"""

# TODO: Check if there is other tags needed? e.g. <CONN>

from parserlib import segment

seg = segment.Segment(stopword_file='external/stop_words.big.txt',
                      dict='external/dict.txt.big')

def read(filename, train):
    result = []
    with open(filename, 'r') as file:
        header = file.readline()[:-1].split(',')
        if train and len(header) != 4:
            print ('Training file with header: ', header, '... please check!')
            return None
        if not train and len(header) != 3:
            print ('Testing file with header: ', header, '... please check!')
            return None

        for line in file.readlines():
            segs = line[:-1].split(',')
            par1 = [ "<ARG1>" ] + seg.parse(segs[1]) + [ "</ARG1>" ]
            par2 = [ "<ARG2>" ] + seg.parse(segs[2]) + [ "</ARG2>" ]

            # For balancing, there are 2 combined data.
            if train:
                result.append((par1 + par2, segs[3]))
                result.append((par1 + par2, segs[3]))
                result.append((par1, segs[3]))
                result.append((par2, segs[3]))
            else:
                result.append((par1 + par2, -1))

    return result
    



