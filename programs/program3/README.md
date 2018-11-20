# Parallel Computing Program 2 -- MPI

A distributed memory program to solve the n-queens problem.

## Program Description

@todo Briefly outline the problem, solution, and program.

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

@todo Get usage nailed down.

## Program Output

@todo Get output format nailed down.

## Solution Strategies

@todo Outline different solution strategies.
