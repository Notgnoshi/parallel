#!/usr/bin/env python3
import subprocess
import numpy as np
import itertools


def perform_timing(procs, runs, n):
    times = []
    for _ in range(runs):
        output = subprocess.check_output([
            'mpiexec', '-np',
            str(procs), 'build/prog3', '--strategy', '3', '--time',
            str(n)
        ]).decode('utf-8')

        lines = output.split('\n')
        for line in lines:
            if 'Elapsed' in line:
                time = float(line.split()[-1][:-1])
                times.append(time)
    times = np.array(times)
    avg = times.mean()
    std = times.std()
    return avg, std


def main():
    """Time the program multiple times."""
    sizes = range(10, 11)
    procs = [32, 48, 64, 72, 96]

    for size, proc in itertools.product(sizes, procs):
        avg, std = perform_timing(proc + 1, 2, size)
        print('='*80)
        print('size:', size)
        print('procs:', proc)
        print('avg:', avg)
        print('std:', std)
        print('='*80)


if __name__ == '__main__':
    main()
