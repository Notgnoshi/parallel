# Parallel Computing Program 3 -- MPI

A distributed memory program to solve the n-queens problem.

## Program Description

This program is a distributed approach to a brute force solution to the @f$n@f$-Queens
problem.

## Acquiring the Program

Clone my program with the following

```shell
# Either clone from github
git clone https://github.com/Notgnoshi/parallel.git
# or from gitlab
git clone https://gitlab.mcs.sdsmt.edu/7314636/parallel.git
```

Everything from here on assumes `.../program3` is the current working directory

```shell
cd parallel/programs/program3
```

Note that all program artifacts (documentation, temporary files, object files, etc)
will be placed in `build/`.

## Documentation

The documentation is buildable by the following

```bash
# Make the HTML and LaTeX source
make docs
# Open the HTML docs in your browser
make viewdocs
# Make the PDF documentation
cd build/latex
make
# Open PDF in default viewer
xdg-open refman.pdf
cd ../../
```

However, note that the LaTeX documentation will not build on the Opp Lab machines.
The HTML documentation can be viewed with `make viewdocs`.

## Unit Tests

I switched to using [Catch2](https://github.com/catchorg/Catch2) for unit tests,
so that I can remove any dependencies that you might have to download, because
Catch2 is a single headerfile that I have included in my repository.

Running `make -j 8 runtests` will build and run the unit tests.

## Compilation and Usage

Run `make -j 8` to build the program. Then it is usable like so

```text
program3 $ build/prog3 --help
Usage: build/prog3 [--help] [--output OUTPUT] [--strategy STRATEGY] [--time] [--verbose] <n>

A distributed memory solution to the n queens problem.

positional arguments:
  n               The length of one side of the chess board.


optional arguments:
 -h, --help       Show this help message and exit
 -o, --output     The filename to print solutions to.
 -v, --verbose    Increase output verbosity.
 -s, --strategy   The strategy to use for the solution. Must be one of
                      1 - Use a slow serial solution strategy.
                      2 - Use a slow shared memory parallel strategy.
                      3 - Use a distributed memory strategy.
 -t, --time       Whether or not to time the solution duration. Will disable I/O.
```

Note that this program implements several solution strategies.

1. A *slow* serial solution.
2. A slightly faster OpenMP shared memory parallel solution.
3. A distributed MPI solution.

Also note that if an output filename is given, each slave process will output its
own file to avoid resource contention.

## Parallel Design

This is a simply problem, with relatively few primitive tasks. Here are a few observations:

* There is a linear method to find the @f$n@f$th permutation of a list of numbers.
* C++ has a `std::next_permutation` function in the algorithm library.
* The combination of the above makes this problem embarrassingly parallel, because
  each slave process needs *no* communication to perform its work.
* There's a fairly simple method to checking if one of the @f$(1, 2, \dots, n)@f$
  tuples is a solution.

There are, of course, mathematical ways to prune the number of solutions so that
the program finishes in a fraction of a second on a single core with @f$n@f$ as large as 14.
However, this assignment is intended to work with distributed computation, so that's
the spirit I wrote it with.

### Partitioning

The computation can be divided as finely as a single permutation being a single task.
However, this does not scale with the problem size very well.

I divided the @f$0, \dots, n!@f$ permutations of @f$(0, \dots, n-1)@f$ evenly among
the number of workers, and each worker checks each permutation it is given. An improvement
I could have made was to prune future permutations based on the current permutation.
This could prune as much of half the number of permutations left each iteration.

### Comunication

There is very little communication to be done, because I chose a static work allocation.
Each worker knows its rank and the problem size, so it can generate its starting
permutation and use `std::next_permutation` to continue until it has finished.

### Agglomeration

As discussed, I divided the permutations evenly among the workers. This eliminated
a lot of communication, and made the implementation much simpler.

### Mapping

My original design had an abstract base `Process` class that `MasterProcess` and `SlaveProcess`
both inherited from. The original intent was for `MasterProcess` to assign work and
receive results from `SlaveProcess`, but the distinction was meaningless.

## Results

All timings were done with the following script

```python
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
    # Edit to change config
    sizes = range(6, 13)
    procs = [1, 2, 4, 16, 64, 128]

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
```

Here is a table of my run times in seconds for different problem sizes and number of processors.
Note that some of the Opp Lab machines were turned off, so I could not utilize all 128 cores.

|@f$n@f$|@f$p=64@f$|@f$p=96@f$|
|:--:|:------:|:------:|
| 10 | 0.0399 | 0.0317 |
| 11 | 0.0436 | 0.0386 |
| 12 |  0.380 |  0.312 |
| 13 |  4.368 |  2.853 |
| 14 | 61.266 | 41.956 |
| 15 | 882.98 | 620.32 |

I think I'll go out on a limb and avoid doing sequential timings on these problem sizes.
So here are timings in seconds for smaller problem sizes.

|@f$n@f$| @f$p=1@f$ | @f$p=2@f$ | @f$p=4@f$ | @f$p=16@f$ | @f$p=64@f$ |
|:--:|:---------:|:---------:|:---------:|:----------:|:----------:|
|  6 | 0.000131  | 0.000228  | 0.000257  | 0.000440   | 0.0351     |
|  7 | 0.000223  | 0.000315  | 0.000319  | 0.000860   | 0.0263     |
|  8 | 0.000678  | 0.000499  | 0.000476  | 0.000437   | 0.0208     |
|  9 | 0.00490   | 0.00289   | 0.00205   | 0.00790    | 0.0428     |
| 10 | 0.0510    | 0.0358    | 0.0227    | 0.00722    | 0.0193     |
| 11 | 0.586     | 0.295     | 0.217     | 0.105      | 0.0521     |
| 12 | 7.2842    | 3.893     | 2.073     | 0.996      | 0.3768     |
| 13 | 98.961    | 51.0267   | 27.64425  | 13.78065   | 4.172065   |

The following python script calculates the relevant metrics

```python
#!/usr/bin/env python3
import numpy as np

procs = [1, 2, 4, 16, 64]
times = [
    [0.000131, 0.000228, 0.000257, 0.000440, 0.0351],
    [0.000223, 0.000315, 0.000319, 0.000860, 0.0263],
    [0.000678, 0.000499, 0.000476, 0.000437, 0.0208],
    [0.00490, 0.00289, 0.00205, 0.00790, 0.0428],
    [0.0510, 0.0358, 0.0227, 0.00722, 0.0193],
    [0.586, 0.295, 0.217, 0.105, 0.0521],
    [7.2842, 3.893, 2.073, 0.996, 0.3768],
    [98.961, 51.0267, 27.64425, 13.780, 4.172],
]

speedups = []

for size in times:
    speedup = []
    seq = size[0]
    for time in size[1:]:
        speedup.append(seq / time)
    speedups.append(speedup)

speedups = np.array(speedups)
print(speedups)

efficiencies = []

for size in times:
    efficiency = []
    seq = size[0]
    for time, p in zip(size[1:], procs[1:]):
        efficiency.append( seq / (time * p))
    efficiencies.append(efficiency)
efficiencies = np.array(efficiencies)
print(efficiencies)

# Take advantage of downwards broadcasting
kfs = (1 / speedups - 1 / np.array(procs[1:])) / (1 - 1 / np.array(procs[1:]))
print(kfs)
```

### Speedup

|@f$n@f$| @f$p=2@f$ | @f$p=4@f$ | @f$p=16@f$ | @f$p=64@f$|
|:--:|:-----:|:-----:|:-----:|:-----:|
|  6 | 0.574 | 0.509 | 0.297 | 0.003 |
|  7 | 0.707 | 0.699 | 0.259 | 0.008 |
|  8 | 1.358 | 1.424 | 1.551 | 0.032 |
|  9 | 1.695 | 2.390 | 0.620 | 0.114 |
| 10 | 1.424 | 2.246 | 7.063 | 2.642 |
| 11 | 1.986 | 2.700 | 5.580 | 11.247 |
| 12 | 1.871 | 3.513 | 7.313 | 19.331 |
| 13 | 1.939 | 3.579 | 7.181 | 23.720 |

### Efficiency

|@f$n@f$| @f$p=2@f$ | @f$p=4@f$ | @f$p=16@f$| @f$p=64@f$|
|:--:|:-----:|:-----:|:------:|:---------:|
|  6 | 0.287 | 0.127 | 0.0186 | 0.0000583 |
|  7 | 0.353 | 0.174 | 0.0162 | 0.000132  |
|  8 | 0.679 | 0.356 | 0.0969 | 0.000509  |
|  9 | 0.847 | 0.597 | 0.0387 | 0.00178   |
| 10 | 0.712 | 0.561 | 0.441  | 0.0412    |
| 11 | 0.993 | 0.675 | 0.348  | 0.175     |
| 12 | 0.935 | 0.878 | 0.457  | 0.302     |
| 13 | 0.969 | 0.894 | 0.448  | 0.370     |

### Karp-Flatt

|@f$n@f$| @f$p=2@f$      | @f$p=4@f$      | @f$p=16@f$     | @f$p=64@f$     |
|:--:|:--------------:|:--------------:|:--------------:|:--------------:|
|  6 | 2.48091603e+00 | 2.28244275e+00 | 3.51603053e+00 | 2.72176057e+02 |
|  7 | 1.82511211e+00 | 1.57399103e+00 | 4.04693572e+00 | 1.19793366e+02 |
|  8 | 4.71976401e-01 | 6.02753196e-01 | 6.20845624e-01 | 3.11495528e+01 |
|  9 | 1.79591837e-01 | 2.24489796e-01 | 1.65306122e+00 | 8.85746680e+00 |
| 10 | 4.03921569e-01 | 2.60130719e-01 | 8.43398693e-02 | 3.68565204e-01 |
| 11 | 6.82593857e-03 | 1.60409556e-01 | 1.24459613e-01 | 7.44460697e-02 |
| 12 | 6.88888279e-02 | 4.61180821e-02 | 7.91832551e-02 | 3.66764683e-02 |
| 13 | 3.12486737e-02 | 3.91265246e-02 | 8.18632256e-02 | 2.69541809e-02 |
