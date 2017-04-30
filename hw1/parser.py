from parserlib.segment import Segment
import argparse
import sys

# punct = u''':!),.:;?]}¢'"、。〉》」』】〕〗〞
         # ︰︱︳﹐､﹒﹔﹕﹖﹗﹚﹜﹞！），．：；？｜｝︴︶︸︺︼︾﹀﹂
         # ﹄﹏､～￠々‖•·ˇˉ―--′’”([{£¥'"‵〈《「『【〔〖（［｛￡￥〝︵
         # ︷︹︻︽︿﹁﹃﹙﹛﹝（｛“‘-—_…'''

punct = u",.，。～；!！"

def parse_args():
    parser = argparse.ArgumentParser(description='Parse paragraphs.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-a', '--aspect', action='store_true',
                       help='Indicate input file as aspect_review.')
    group.add_argument('-p', '--polarity', action='store_true',
                       help='Indicate input file as polarity_review.')
    group.add_argument('-t', '--test', action='store_true',
                       help='Indicate input file as test_review.')

    parser.add_argument('-b', '--bigram', action='store_true',
                        help='Use bigram for parsing')

    parser.add_argument('-n', '--no-trim', action='store_true',
                        help='No trim the punctuations in paragraphs.')
    parser.add_argument('-i', '--input', required=True,
                        help='Indicate input file name.')
    parser.add_argument('-o', '--output', help='Indicate output file name.')

    return parser.parse_args(sys.argv[1:])

def parse_out(seg, line, output, trim):
    output.write(" ".join(seg.parse(line[:-1], trim)) + "\n")

def bigram(line, output, trim):
    ll = []
    for i in range(len(line[:-1])):
        if line[i+1] in punct:
            continue
        elif line[i] in punct:
            ll.append(line[i] if trim else " ")
        else:
            ll.append(line[i:i+2])
    
    output.write(" ".join(ll) + "\n")

if __name__ == '__main__':
    opt = parse_args()
    output = sys.stdout
    if opt.output:
        output = open(opt.output, "w")
    trim = not opt.no_trim

    seg = Segment(dict="external/dict.txt.big")

    with open(opt.input, "r") as file:
        line = file.readline()
        while line:
            if opt.aspect:  # aspect_review
                output.write(line)  # review index
                line = file.readline()
                # Outputs parsed single line.
                if opt.bigram:
                    bigram(line, output, trim)
                else:
                    parse_out(seg, line, output, trim)  # review content
                line = file.readline()
                output.write(line)  # positive aspect 
                line = file.readline()
                output.write(line)  # negative aspect 

            elif opt.polarity:  # polarity_review
                num, str = line.split('\t', 1)  # Maxsplit: 1
                output.write(num + '\t')
                # Outputs parsed single line.
                if opt.bigram:
                    bigram(line, output, trim)
                else:
                    parse_out(seg, str, output, trim)  # review content

            else:  # test_review
                output.write(line)  # review index
                line = file.readline()
                # Outputs parsed single line.
                if opt.bigram:
                    bigram(line, output, trim)
                else:
                    parse_out(seg, line, output, trim)  # review content
        
            # Next new start.
            line = file.readline()

