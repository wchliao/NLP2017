from parserlib import segment

seg = segment.Segment(stopword_file='./discourse-ablstm/external/stop_words.big.txt',
                      dict='./discourse-ablstm/external/dict.txt.big')

def parse(paragraph):
    return filter(lambda x: x and len(x) > 0 and x != " ", seg.parse(paragraph))


def parseTrain(input, output):
    output = open(output, 'w')

    with open(input, 'r') as file:
        output.write(file.readline())

        for line in file.readlines():
            id, seg1, seg2, tag = line[:-1].split(',')
            output.write(id + ",")
            output.write(" ".join(parse(seg1)) + ",")
            output.write(" ".join(parse(seg2)) + ",")
            output.write(tag + "\n")

    output.close()


def parseTest(input, output):
    output = open(output, 'w')

    with open(input, 'r') as file:
        output.write(file.readline())

        for line in file.readlines():
            id, seg1, seg2 = line[:-1].split(',')
            output.write(id + ",")
            output.write(" ".join(parse(seg1)) + ",")
            output.write(" ".join(parse(seg2)) + "\n")

    output.close()

