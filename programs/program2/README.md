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
