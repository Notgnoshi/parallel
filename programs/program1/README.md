# Program 1

---

Clone and build my projects with the following

```bash
# Either clone from github
git clone https://github.com/Notgnoshi/parallel.git
# or from gitlab
git clone https://gitlab.mcs.sdsmt.edu/7314636/parallel.git
cd parallel/programs/program1
# Make both programs
make
# Make the HTML and LaTeX source
make docs
# Open the HTML docs in your browser
make viewdocs
cd docs/latex
# Make this PDF
make
xdg-open refman.pdf
cd ../../
# Build artifacts placed in ./build/
cd build
# Run programs
./circuit 1
./circuit 2
./prime 10
```

---

Note that both programs are in the same source tree, and compiled with the same
makefile. However, they must be compiled with different targets as my makefile
magic is slightly lacking when running on so little sleep.

This documentation can be built and viewed by running `make docs` and then running
`make viewdocs` to open the HTML documentation in your default web browser.

Both programs may be compiled at once by simply running `make`. However, since my
makefile magic is mediocre, the dependency tree for both targets is not properly
computed by make, so if any changes are made to one of the programs, the corresponding
target must be made individually.

## Circuit Satisfiability

This program brute force checks a given digital circuit, represented as a complex
boolean expression, for its satisfiability. That is, it iterates over every
possible input and records which inputs result in the expression evaluating to
`true`.

Due to the lack of data dependency, this makes a very nice candidate for parallelization.
This program parallelizes on @f$p@f$ threads, where @f$p@f$ is the number of processors
on the system. Due to the constant nature of each computation, the default implementation
will use a static work schedule.

However, if compiled with `-DSCHEDULE_COMPARISON`, the default, static, and dynamic
schedules will be timed and compared. The static and dynamic schedules use a chunk
size of 1 in the comparison, while the default schedule splits the work evenly among
the workers in contiguous chunks.

The provided makefile defines `SCHEDULE_COMPARISON` by default.

Compile by running `make circuit`. All build artifacts will be placed in `./build/`.
Run the program with `build/circuit <n>` where `<n>` is either `1` or `2`.

## Prime Sieve

This program computes all prime numbers less than some given limit using the Sieve
of Eratosthenes.

The Sieve of Eratosthenes works by this simple observation: Determining whether a
number is prime or not is *hard*. It's far easier to mark off numbers we know are
*not* prime.

So we allocate a large boolean array initialized to all `true` (indicating that
each index is prime) and then for each index, if the index is marked as prime, mark
all multiples of that index as not prime (`false`).

This program does not respect the `-DSCHEDULE_COMPARISON` compilation definition.
Because, unlike the first program, this is a computationally intensive program. Instead,
I manually compiled and ran with a static and dynamic work schedule, noting that
the static schedule was noticeably slower. This is because each iteration takes a
non-constant amount of work, so a dynamic schedule is preferable.

However, note that there is repeated work, because suppose thread 4 starts its work
on index `55` before any of the other threads mark `55` as not prime. Then thread 4
will do extra work, that other threads will duplicate. (contrived example.) This is
one of two reasons I believe my parallel implementation isn't as fast as I would have
liked. The other reason is that I think the larger sieves don't fit into the cache,
so there's isn't as much (a few seconds) speedup between the parallel and serial
implementation.

There are a few ways to improve this implementation that I can think of. First, and
easiest, don't even bother checking even numbers. You can index your sieve array
in such a way to half the memory usage by constructing a mapping
@f$\{3, 5, 7, \dots, n\} \to \{0, 1, 2, \dots, m\}@f$ by index with `j / 2`. This
will reduce the memory usage of the program by a half, but complicates the retrieval
of the prime numbers given the sieve array.

The next optimization would be to use a more efficient data packing scheme than
using an entire byte to store `true` or `false`. I'm entirely unwilling to write my
own, but using the C++ `vector<bool>` specialization would help considerably with
memory usage.

The last optimization attempts to tackle the cache misses I believe (but don't have
proof) that I'm victim of. I think there should be a way to split the problem into
blocks, each of which should fit in the cache. Then the outer loop should iterate
as usual, but the inner loop should be restricted to the assigned block of the
sieve array. Then each worker can perform its work on a piece of the problem without
forcing loading and unloading from the cache.

As with the previous program, running `make prime` will place the `prime` build
artifact in `./build/`. Run the program with `build/prime <n>` where `<n>` is the
upper limit on your prime sieve. If enough memory cannot be allocated, a helpful
error message is printed to `stderr` and the program is shut down.
