# Parallel Computing Program 2 -- CUDA

A CUDA-accelerated program to compute matrix-vector multiplication and matrix-matrix addition.

## Compilation

* `make` will build the `./build/prog2` executable.
* `make runtests` will build and run the `./build/testsuite` executable.
* `make viewdocs` will build and open this documentation.
* `make profile` will build and profile the program against some fixed input with `nvprof`.

## Usage

The program is used as follows

```text
Usage: build/prog2 [--help] [--output] [--kernel] <operation> <input1> <input2>

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

@todo Implement the kernels on the CPU first
@todo Implement several kernels for each and give the option of choosing which one to use
@todo save results to a file?
@todo Make a `profile` target that profiles with `nvprof  --unified-memory-profiling off`
@todo Unit tests!

## Notes
