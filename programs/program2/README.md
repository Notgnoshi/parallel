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
$ ./pack.py tests/matrices/4x4_seq.txt tests/matrices/4x4_seq.mat
Input file dimensions:  4 x 4
$ hexdump tests/matrices/4x4_seq.mat
0000000 0004 0000 0000 0000 0004 0000 0000 0000
0000010 0000 0000 0000 3ff0 0000 0000 0000 4000
0000020 0000 0000 0000 4008 0000 0000 0000 4010
0000030 0000 0000 0000 4014 0000 0000 0000 4018
0000040 0000 0000 0000 401c 0000 0000 0000 4020
0000050 0000 0000 0000 4022 0000 0000 0000 4024
0000060 0000 0000 0000 4026 0000 0000 0000 4028
0000070 0000 0000 0000 402a 0000 0000 0000 402c
0000080 0000 0000 0000 402e 0000 0000 0000 4030
0000090
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
0000010 5844 3a0c 2605 3fe6 4ac0 ee56 e5f2 3fa9
0000020 57ec e2ea 1d91 3fe0 218b efef f40d 3fe8
0000030 e30f c2a0 d630 3fe6 9be5 7ec8 ea6b 3fed
0000040 1ab8 c97a 004d 3fb8 8830 4eb5 af69 3fab
0000050 04a0 1f7d 3413 3fda d01f b24c cec5 3fef
0000060 017a af05 11b2 3fe4 0ec8 c5bc 65bb 3fb4
0000070 4108 3761 9fdf 3fe0 15a0 1b87 b792 3fa6
0000080 62af 2add bc8a 3fe1 546a e651 5528 3fef
0000090
```

## Profiling

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
| M+M | 25.42% (21.836ms) | 7.36% (5.1599ms)  | 2.54% (1.6991ms)  | 2.13% (1.4148ms)  | 2.09% (1.3776ms)  | 2.07% (1.3548ms)  |
| M*V | 29.10% (9.2326ms) | 18.93% (5.2198ms) | 12.02% (3.0329ms) | 7.37% (1.8300ms)  | 12.17% (3.1426ms) | 21.84% (6.2511ms) |
| M*M | 99.81% (35.0649s) | 99.26% (8.68235s) | 97.64% (2.68512s) | 91.90% (740.97ms) | 91.65% (719.67ms) | 91.84% (731.49ms) |

where the table heading is the dimension of the square blocks, and the rows are
the type of operation being performed.

As expected, smaller block sizes perform worse, with the exception of matrix-vector
multiplication, which performs best with @f$8 \times 8@f$ block sizes. This must
be the point at which the benefit of shared memory and the cost of wasteful blocks
meet.

## Matrix Multiplication

See `#MultiplicationKernel` for more complete details, but I ended up implementing
full matrix-matrix multiplication because I wanted to use shared memory, but for
non-square block sizes this really sucks (I couldn't find a way to do it). If the
block sizes are square, then it's only a little bit more complicated to implement
full multiplication.
