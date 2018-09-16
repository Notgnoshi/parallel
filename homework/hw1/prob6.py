#!/usr/bin/env python3
import itertools

def print_bits(n):
    """Print the lower 4 bits of `n`"""
    mask = 1 << 3
    while mask:
        print(int(bool(n & mask)), end='')
        mask = mask >> 1


def rotl4(x, n=1):
    """Rotates x left by n bits"""
    return 0b00001111 & ((x << n) | (x >> (4 - n)))


def tikzify(exchanges, shuffles):
    """Generates the tedious part of the LaTeX Tikz figure"""
    for a in range(16):
        a = bin(a)[2:].zfill(4)
        print(f'\\node[vertex] ({a}) {{${a}$}};')

    print()

    for u, v in exchanges:
        u = bin(u)[2:].zfill(4)
        v = bin(v)[2:].zfill(4)
        print(f'\\draw[->] ({u}) edge node[]{{E}} ({v});')

    print()

    for u, v in shuffles:
        u = bin(u)[2:].zfill(4)
        v = bin(v)[2:].zfill(4)
        print(f'\\draw[->] ({u}) edge node[]{{S}} ({v});')


def main():
    """For all pairs of 4 bit numbers, label exchange and shuffle edges"""
    # The exchange edges are duplicated, one going in each direction
    exchanges = []
    # The shuffle edges go from the first to the second node
    shuffles = []

    # Compair every pair of nodes
    for u, v in itertools.product(range(16), repeat=2):
        # If the addresses differ only in the last bit, they form an exchange edge
        if u == (v ^ 0b0001):
            exchanges.append((u, v))

        # If u, rotate left 1 bit is equal to v, the edge u -> v is a shuffle edge
        if rotl4(u) == v:
            shuffles.append((u, v))

    tikzify(exchanges, shuffles)


if __name__ == '__main__':
    main()
