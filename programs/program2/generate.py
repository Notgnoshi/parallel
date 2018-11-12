#!/usr/bin/env python3
import argparse
from struct import pack

import numpy as np

DESCRIPTION = 'Generate random matrices.'
VERSION = '0.1'


def parse_args():
    """Parse commandline arguments."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--version', '-v', version=VERSION, action='version')
    parser.add_argument('rows', type=int, help='The height of the matrix to generate')
    parser.add_argument('cols', type=int, help='The width of the matrix to generate')
    parser.add_argument('--output', '-o', default='matrix.mat',
                        help='The output filename. Defaults to `matrix.mat`')

    return parser.parse_args()


def main(args):
    """Generate a random matrix and writes it to a binary file.

    :param args: The commandline arguments.
    """
    M = np.random.rand(args.rows, args.cols)

    with open(args.output, 'wb') as binary:
        binary.write(pack('N', args.rows))
        binary.write(pack('N', args.cols))

        for row in M:
            binary.write(pack('%sf' % len(row), *row))



if __name__ == '__main__':
    main(parse_args())
