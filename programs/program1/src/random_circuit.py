#!/usr/bin/env python3
import random


def gen_inputs(n, limit=16):
    """Generates n random numbers between 0 and limit (exclusive)"""
    for _ in range(n):
        yield random.randrange(limit)


def gen_not():
    """Randomly returns the string '!'"""
    return '!' if random.randint(0, 1) else ''


def gen_sum(length=2):
    """Generates a single random sum of the given length."""
    VAR = 'bits'
    terms = []
    for term in gen_inputs(length):
        terms.append(gen_not() + VAR + f'[{term}]')

    return f'({" || ".join(terms)})'


def main():
    """Generates a random 16 input digital circuit in product of sum form."""
    terms = []
    for _ in range(18):
        terms.append(gen_sum(2))

    print(' && '.join(terms))


if __name__ == '__main__':
    main()
