# Parallel Computing Program 2 -- CUDA

A CUDA-accelerated program to compute matrix-vector multiplication and matrix-matrix addition.

## Compilation

* `make` will build the `./build/prog2` executable.
* `make runtests` will build and run the `./build/testsuite` executable. Note
   some of the unit tests rely on file paths that are relative to the source tree,
   so do not run the unit tests with `./testsuite`; be sure to use the `make runtests`
   target in the program root directory.
* `make viewdocs` will build and open this documentation.
* `make profile` will build and profile the program against some fixed input with `nvprof`.

## Usage

The program is used as follows

```text
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
                   0 - Use the default kernel for the given operation.
                   1 - Do not use a CUDA kernel. Perform the operation on the CPU.
```

Example matrix and vector files will be provided.

@todo Make a `profile` target that profiles with `nvprof  --unified-memory-profiling off`

## Notes

I chose to read matrices/vector from files for the most flexibility. However, I
chose a file format that would make `memcpy`ing the binary file into a struct the
easiest. Unfortunately, this makes creating the files slightly unwieldy. I have
included a Python script `pack.py` to convert human-readable text matrix files
into the binary format my program expects. For example,

```bash
$ cat tests/matrices/4x4_seq.txt
1  2  3  4
5  6  7  8
9  10 11 12
13 14 15 16
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

For details, see the `Matrix_t` struct documentation and implementation.
