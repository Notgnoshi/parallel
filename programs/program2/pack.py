#!/usr/bin/env python3
import argparse
from struct import pack


DESCRIPTION = 'Pack text file matrices into correct format for program 2.'
VERSION = '0.1'


def parse_args():
    """Parse commandline arguments for this script."""
    parser = argparse.ArgumentParser(description=DESCRIPTION)

    parser.add_argument('--version', '-v', version=VERSION, action='version')
    parser.add_argument('input',
                        help='The space-separated text input file. '
                             'Does not store dimensions; the dimensions are '
                             'determined by the file format.')
    parser.add_argument('output',
                        help='The output binary file to pack into. '
                             'The given filename will be overwritten if it already '
                             'exists.')

    return parser.parse_args()


def blocks(file, size=65536):
    """Read a large text file block by block.

    :param file: The file object to read from.
    :param size: The block size, defaults to 65536
    """
    while True:
        block = file.read(size)
        if not block:
            break
        yield block


def get_dimensions(filename):
    """Get the dimensions of the matrix in the given file.

    No error checking is done, so the matrix file *must* be valid (no missing
    data, all rows same length, etc.)

    :param filename: The filename of the matrix to get the dimensions from.
    """
    rows = 0
    cols = 0
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        # Get the number of columns from the first row.
        row = f.readline().strip().split()
        cols = len(row)
        if row and cols:
            # Count the row already read.
            rows = 1
            # Iterate over the file in blocks
            rows += sum(block.count('\n') for block in blocks(f))

    return rows, cols


def pack_into(input_filename, output_filename):
    """Pack the given matrix into the given binary filename.

    :param input_filename: The input filename.
    :param output_filename: The output filename.
    """
    rows, cols = get_dimensions(input_filename)
    # print('Input file dimensions: ', rows, 'x', cols)

    with open(input_filename, 'r') as text, open(output_filename, 'wb') as binary:
        binary.write(pack('N', rows))
        binary.write(pack('N', cols))

        for row in text:
            row = [float(x) for x in row.strip().split()]
            binary.write(pack('%sf' % len(row), *row))


def main(args):
    """Convert text matrix files to binary format.

    :param args: The commandline arguments for this script.
    """
    pack_into(args.input, args.output)


if __name__ == '__main__':
    main(parse_args())
