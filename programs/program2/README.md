# Parallel Computing Program 2 -- CUDA

A CUDA-accelerated program to compute matrix-vector multiplication and matrix-matrix addition.

## Acquiring the Program

Clone my program with the following

```shell
# Either clone from github
git clone https://github.com/Notgnoshi/parallel.git
# or from gitlab
git clone https://gitlab.mcs.sdsmt.edu/7314636/parallel.git
```

Everything from here on assumes `.../program2` is the current working directory

```shell
cd parallel/programs/program2
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

## Unit Tests

The unit tests have a dependency on [CppUnit](https://freedesktop.org/wiki/Software/cppunit/).
On Ubuntu, CppUnit is easily installable with

```shell
sudo apt install libcppunit-dev
```

Then running `make runtests` will build and run the unit tests. Note that some
of the tests rely on file paths that are relative to the source tree. So either
run `./build/testsuite` or `make runtests` to run the unit tests.

## Compilation and Usage

The program is buildable with `make`, which will produce the `build/prog2` executable.

The program is used as follows

```shell
$ build/prog2 --help
Usage: build/prog2 [--help] [--output <file>] [--kernel] <operation> <input1> <input2>

CUDA accelerated matrix operations.

positional arguments:
  operation    One of {MVM, MMA}
  input1       An input file for the left operand
  input2       An input file for the right operand


optional arguments:
 -h, --help    Show this help message and exit
 -o, --output  Save the output to the given file
 -k, --kernel  The kernel to use for the given operation. Must be one of
                   0 - Use the default (CUDA) kernel for the given operation.
                   1 - Do not use a CUDA kernel. Perform the operation on the CPU.
                   2 - Use the CUDA kernel for the given operation.
```

Example matrix and vector files will be provided.

## Input File Format

I chose to read matrices/vector from files for the most flexibility. However, I
chose a file format that would make `memcpy`ing the binary file into a struct the
easiest. Unfortunately, this makes creating the files slightly unwieldy. I have
included two Python scripts `generate.py` and `pack.py` to generate random matrices,
and to convert human-readable text matrix files into the binary format my program
expects. For example, provided the `tests/matrices/4x4_seq.txt` matrix

```shell
$ cat tests/matrices/4x4_seq.txt
1  2  3  4
5  6  7  8
9  10 11 12
13 14 15 16
```

You can convert this matrix into the binary format like so:

```shell
$ ./pack.py --help
usage: pack.py [-h] [--version] input output

Pack text file matrices into correct format for program 2.

positional arguments:
  input          The space-separated text input file. Does not store
                 dimensions; the dimensions are determined by the file format.
  output         The output binary file to pack into. The given filename will
                 be overwritten if it already exists.

optional arguments:
  -h, --help     show this help message and exit
  --version, -v  show programs version number and exit
$ ./pack.py tests/matrices/4x4_seq.txt build/tmp/4x4_seq.mat
Input file dimensions:  4 x 4
$ hexdump build/tmp/4x4_seq.mat
0000000 0004 0000 0000 0000 0004 0000 0000 0000
0000010 0000 3f80 0000 4000 0000 4040 0000 4080
0000020 0000 40a0 0000 40c0 0000 40e0 0000 4100
0000030 0000 4110 0000 4120 0000 4130 0000 4140
0000040 0000 4150 0000 4160 0000 4170 0000 4180
0000050
```

For details on the file format, see the `Matrix_t` struct documentation and implementation.

Writing a text file and converting it into a binary format is time consuming, so
there is also the `generate.py` script to generate random matrices with entries
pulled from a standard normal distribution. This script is dependent on `numpy`.

```shell
$ ./generate.py --help
usage: generate.py [-h] [--version] [--output OUTPUT] rows cols

Generate random matrices.

positional arguments:
  rows                  The height of the matrix to generate
  cols                  The width of the matrix to generate

optional arguments:
  -h, --help            show this help message and exit
  --version, -v         show programs version number and exit
  --output OUTPUT, -o OUTPUT
                        The output filename. Defaults to 'matrix.mat'
$ ./generate.py --output build/tmp/example.mat 4 4
$ hexdump build/tmp/example.mat
0000000 0004 0000 0000 0000 0004 0000 0000 0000
0000010 77cd 3f66 e601 3f44 a4f7 3f62 2ab7 3e47
0000020 f7be 3eef 4433 3d8f db37 3ed6 32a3 3e7a
0000030 4b0e 3f68 8ef9 3e6f c3cc 3f52 bbe6 3f68
0000040 bb8d 3ee1 f5a1 3ebc 2011 3da1 6611 3f7b
0000050
```

## Profiling

I'm not sure how to increase the number of processors on a graphics card, so I
decided to perform my analysis using different block sizes.

The provided makefile defines a *quite* disgusting `profile` target that runs both
the matrix-matrix addition and matrix-vector multiplication several times with
varying block sizes. If I would have thought ahead, I would have tried to find a
way to allow runtime defined block sizes.

Roughly, the `profile` target does the following

1. Cleans all build artifacts so that we can be sure nothing was built without
   optimization, or with debugging symbols injected.
2. Builds the `build/prog2` executable.
3. Generates several test matrices of a large, fixed size (@f$n = 4000@f$) defined
   by the `_SIZE` variable in the makefile.
4. For both the matrix addition, matrix vector multiplication, and matrix-matrix
   multiplication operations, with block sizes @f$b \in \{1, 2, 4, 8, 16, 32\}@f$ it:
   1. edits the block size with `sed`.
   2. recompiles and links the modified translation unit.
   3. runs the executable with `nvprof`, but with the output piped to `grep` so
      that we only see the duration of the kernel call.
5. Then the block sizes are reset to @f$16 \times 16@f$ for both kernels.

I wasn't even sure if @f$1 \times 1@f$ or @f$2 \times 2@f$ blocks would even work
for multiplication, so run

```shell
$ diff --report-identical-files --from-file build/tmp/addition_prof_result*
Files build/tmp/addition_prof_result16.mat and build/tmp/addition_prof_result1.mat are identical
Files build/tmp/addition_prof_result16.mat and build/tmp/addition_prof_result2.mat are identical
Files build/tmp/addition_prof_result16.mat and build/tmp/addition_prof_result32.mat are identical
Files build/tmp/addition_prof_result16.mat and build/tmp/addition_prof_result4.mat are identical
Files build/tmp/addition_prof_result16.mat and build/tmp/addition_prof_result8.mat are identical
$ diff --report-identical-files --from-file build/tmp/multiplication_prof_result*
Files build/tmp/multiplication_prof_result16.mat and build/tmp/multiplication_prof_result1.mat are identical
Files build/tmp/multiplication_prof_result16.mat and build/tmp/multiplication_prof_result2.mat are identical
Files build/tmp/multiplication_prof_result16.mat and build/tmp/multiplication_prof_result32.mat are identical
Files build/tmp/multiplication_prof_result16.mat and build/tmp/multiplication_prof_result4.mat are identical
Files build/tmp/multiplication_prof_result16.mat and build/tmp/multiplication_prof_result8.mat are identical
$ diff --report-identical-files --from-file build/tmp/extended_multiplication_prof_result*
Files build/tmp/extended_multiplication_prof_result16.mat and build/tmp/extended_multiplication_prof_result1.mat are identical
Files build/tmp/extended_multiplication_prof_result16.mat and build/tmp/extended_multiplication_prof_result2.mat are identical
Files build/tmp/extended_multiplication_prof_result16.mat and build/tmp/extended_multiplication_prof_result32.mat are identical
Files build/tmp/extended_multiplication_prof_result16.mat and build/tmp/extended_multiplication_prof_result4.mat are identical
Files build/tmp/extended_multiplication_prof_result16.mat and build/tmp/extended_multiplication_prof_result8.mat are identical
```

to verify that each file is identical.

On my machine (GTX 1080, with 8GB video ram)

```text
$ screenfetch
                          ./+o+-       nots@abyss
                  yyyyy- -yyyyyy+      OS: Ubuntu 18.04 bionic
               ://+//////-yyyyyyo      Kernel: x86_64 Linux 4.15.0-38-generic
           .++ .:/++++++/-.+sss/`      Uptime: 7d 4h 45m
         .:++o:  /++++++++/:--:/-      Packages: 3162
        o:+o+:++.`..```.-/oo+++++/     Shell: bash 4.4.19
       .:+o:+o/.          `+sssoo+/    Resolution: 7680x2160
  .++/+:+oo+o:`             /sssooo.   DE: GNOME
 /+++//+:`oo+o               /::--:.   WM: GNOME Shell
 \+/+o+++`o++o               ++////.   WM Theme: Adwaita
  .++.o+++oo+:`             /dddhhh.   GTK Theme: NumixDark [GTK2/3]
       .+.o+oo:.          `oddhhhh+    Icon Theme: Numix-Circle
        \+.++o+o``-````.:ohdhhhhh+     Font: SFNS Display 12
         `:o+++ `ohhhhhhhhyo++os:      CPU: Intel Core i7-6700K @ 8x 4.4GHz [27.8°C]
           .o:`.syhhhhhhh/.oo++o`      GPU: GeForce GTX 1080
               /osyyyyyyo++ooo+++/     RAM: 8098MiB / 15998MiB
                   ````` +oo+++o\:
                          `oo++.
```

the profile results are summarized

|     |         1         |         2         |         4         |         8         |         16        |         32        |
|:---:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|:-----------------:|
| M+M | 40.15% (21.820ms) | 14.17% (5.3356ms) | 4.14%  (1.3969ms) | 2.44%  (803.49us) | 2.11%  (693.25us) | 2.42%  (797.00us) |
| M*V | 39.59% (7.7031ms) | 28.14% (4.4283ms) | 10.97% (1.4275ms) | 8.38%  (1.0299ms) | 7.34%  (902.95us) | 12.55% (1.6458ms) |
| M*M | 99.89% (30.1797s) | 99.52% (6.66770s) | 97.28% (1.14955s) | 92.88% (421.08ms) | 85.69% (195.00ms) | 84.14% (170.29ms) |

where the table heading is the dimension of the square blocks, and the rows are
the type of operation being performed. The percentages are the amount of time spent
in the kernel versus other operations like `cudaMalloc` or `cudaMemcpy`. The times
are the time spent in the kernel call.

As expected, smaller block sizes perform worse, with the exception of matrix-vector
multiplication. This must be the point at which the benefit of shared memory and
the cost of wasteful blocks meet.

While I could measure the amount of time spent in the CPU kernel call (my original plan)
in order to calculate the speedups, this would be comparing apples and oranges
because the timing method differs, I cannot guarantee that the timing method would
be comparable to whatever `nvprof` does under the hood. So I treat the @f$1 \times 1@f$
block times as the sequential run times, even though they *are* parallel.

Then the following python script calculates the speedups, efficiencies, and Karp-Flatt
metrics

```python
#!/usr/bin/env python3
import numpy as np

procs = np.array([1, 2, 4, 8, 16, 32])
mpm_times = np.array([21.820, 5.3356, 1.3969, 0.80349, 0.69325, 0.797])
mtv_times = np.array([7.7031, 4.4283, 1.4275, 1.0299, 0.90295, 1.6458])
mtm_times = np.array([1000*30.1797, 1000*6.6677, 1000*1.14955, 421.08, 195, 170.29])

for times in [mpm_times, mtv_times, mtm_times]:
    serial = times[0]
    speedups = serial / times
    efficiencies = speedups / procs
    kfs = (1 / speedups - 1 / procs) / (1 - 1 / procs)

    print(speedups)
    print(efficiencies)
    print(kfs)
    print('='*80)
```

which outputs

|  1  |      2      |      4      |      8      |      16     |      32     |
|:---:|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| 1   | 4.08951196  | 15.6203021  | 27.15652964 | 31.47493689 | 27.37766625 |
| 1   | 2.04475598  | 3.90507552  | 3.3945662   | 1.96718356  | 0.85555207  |
| NaN | -0.51094409 | -0.24797434 | -0.10077308 | -0.03277727 | 0.00544632  |

for the matrix-matrix addition, and

|  1  |      2     |      4      |      8     |     16     |     32     |
|:---:|:----------:|:-----------:|:----------:|:----------:|:----------:|
| 1   | 1.73951629 | 5.39621716  | 7.47946403 | 8.53103716 | 4.68045935 |
| 1   | 0.86975815 | 1.34905429  | 0.934933   | 0.53318982 | 0.14626435 |
| NaN | 0.14974491 | -0.08624666 | 0.00994219 | 0.05836698 | 0.18828825 |

for matrix-vector multiplication, and

|  1  |      2      |      4      |      8      |      16      |      32      |
|:---:|:-----------:|:-----------:|:-----------:|:------------:|:------------:|
| 1   | 4.52625343  | 26.2534905  | 71.67212881 | 154.76769231 | 177.22532151 |
| 1   | 2.26312672  | 6.56337262  | 8.9590161   | 9.67298077   | 5.5382913    |
| NaN | -0.55813345 | -0.28254644 | -0.12691151 | -0.05977462  | -0.02643351  |

for matrix-matrix multiplication.

Note that the efficiencies and Karp-Flatt metric values are not between 0 and 1,
immediately indicating that something is wrong. This is probably because using
the block size as a stand-in for the number of processors is not valid. I do not
know what else to do though. It might also be that I asserted that the runtime
with a block size of @f$1 \times 1 @f$ was the sequential runtime.

## Matrix Multiplication

See `#MultiplicationKernel` for more complete details, but I ended up implementing
full matrix-matrix multiplication because I wanted to use shared memory, but for
non-square block sizes this really sucks (I couldn't find a way to do it). If the
block sizes are square, then it's only a little bit more complicated to implement
full multiplication.
