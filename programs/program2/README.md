# Parallel Computing Program 2 -- CUDA

A CUDA-accelerated program to compute matrix-vector multiplication and matrix-matrix addition.

## Compilation and Usage

* `make` will build the `./build/prog2` executable.
* `make runtests` will build and run the `./build/testsuite` executable.
* `make viewdocs` will build and open this documentation.
* `make profile` will build and profile the program against some fixed input with `nvprof`.

@todo read in matrices and vectors from the command line (save in struct with size, then just serialize)
@todo Implement the kernels on the CPU first
@todo Implement several kernels for each and give the option of choosing which one to use
@todo save results to a file?
@todo Make a `profile` target that profiles with `nvprof  --unified-memory-profiling off`
@todo Unit tests!

## Notes
