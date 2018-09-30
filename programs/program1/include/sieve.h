/**
 * @file sieve.h
 * @author Austin Gill (atgill@protonmail.com)
 * @brief Usual header file for prime sieve functions.
 */
#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>

/**
 * @brief Allocate and initialize a boolean array for use in a prime sieve.
 *
 * Initializes each element to `true`.
 *
 * If the given length cannot be allocated, print an error and exit the program.
 *
 * @param length The length of the array to allocate and initialize.
 */
bool* allocate( size_t length );

/**
 * @brief Runs the Sieve of Eratosthenes on the given array.
 *
 * Rather than iterating over a large number of numbers and running an expensive
 * primality check on all of them, it's cheaper (computationally) to mark off
 * numbers you know are *not* prime. However, this comes at a large memory cost.
 *
 * Starting with a boolean array initialized to all `true`, iterate over the
 * indices, and if the value stored at an index is `true`, mark all multiples
 * of the index as `false`. This sieves for the primes, hence the name.
 *
 * Since some iterations will take longer than others, parallelize with a dynamic
 * schedule so that workers can get more work once they're finished. (No coffee
 * breaks for you!) Using a static schedule results in slower code, which is
 * expected due to the nature of the problem.
 *
 * @todo I suspect that with large sieves I'm running into cache problems. Find
 *       a way to cut the sieve into blocks that fit into the cache.
 * @todo I'm not getting as much of a speedup as I'd hope for. I think it's probably something to do with cache misses.
 *
 * @param[out] sieve A boolean array indicating wether or not each index is prime
 *             or not. Initialized to `true`.
 * @param length The length of the given array.
 * @returns the time taken to compute the sieve in milliseconds.
 */
double prime_sieve( bool* sieve, size_t length );

/**
 * @brief Print the prime numbers from the given sieve.
 *
 * @param sieve
 * @param length
 */
void print_primes( const bool* sieve, size_t length );
