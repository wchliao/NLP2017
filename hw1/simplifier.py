
import argparse
import sys
import functions as func

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_file', help='give input file here.')
    parser.add_argument('-o', '--output_file', help='give output file here.')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    func.toSimplified(args.input_file, args.output_file)
