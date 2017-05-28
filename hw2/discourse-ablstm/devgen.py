#!/usr/bin/env python

"""
The random generator of dev file.
"""

from __future__ import print_function
import argparse
from random import random
import sys

def parse_args():
    parser = argparse.ArgumentParser(description='Dev file generator')
    parser.add_argument('-i', '--input', required=True, help='Input training file')
    parser.add_argument('-t', '--train-out', required=True, help='Output file of training data')
    parser.add_argument('-d', '--dev-out', required=True, help='Output file of dev data')
    parser.add_argument('-p', '--portion', type=float, default=0.1, help='The portion of dev data')
    return parser.parse_args()

if __name__ == '__main__':
    opt = parse_args()

    train = open(opt.train_out, 'w')
    dev = open(opt.dev_out, 'w')

    with open(opt.input, 'r') as input:
        # Write .csv header to both of the files
        line = input.readline()
        train.write(line)
        dev.write(line)

        for line in input.readlines():
            if random() > opt.portion: # Training data
                train.write(line)
            else:
                dev.write(line)

    train.close()
    dev.close()
                
            
