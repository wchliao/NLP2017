from parserlib.segment import Segment
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Parse paragraphs.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-a', '--aspect', action='store_true',
                       help='Indicate input file as aspect_review.')
    group.add_argument('-p', '--polarity', action='store_true',
                       help='Indicate input file as polarity_review.')
    group.add_argument('-t', '--test', action='store_true',
                       help='Indicate input file as test_review.')

    parser.add_argument('-i', '--input', required=True,
                        help='Indicate input file name.')
    parser.add_argument('-o', '--output', help='Indicate output file name.')

    return parser.parse_args(sys.argv[1:])

def parse_out(seg, line, output):
    output.write(" ".join(seg.parse(line[:-1])) + "\n")


if __name__ == '__main__':
    opt = parse_args()
    output = sys.stdout
    if opt.output:
        output = open(opt.output, "w")

    seg = Segment(dict="external/dict.txt.big")

    with open(opt.input, "r") as file:
        line = file.readline()
        while line:
            if opt.aspect:  # aspect_review
                output.write(line)  # review index
                line = file.readline()
                parse_out(seg, line, output)  # review content
                line = file.readline()
                output.write(line)  # positive aspect 
                line = file.readline()
                output.write(line)  # negative aspect 

            elif opt.polarity:  # polarity_review
                num, str = line.split('\t', 1)  # Maxsplit: 1
                output.write(num + '\t')
                parse_out(seg, str, output)

            else:  # test_review
                output.write(line)  # review index
                line = file.readline()
                parse_out(seg, line, output)  # review content
        
            # Next new start.
            line = file.readline()

