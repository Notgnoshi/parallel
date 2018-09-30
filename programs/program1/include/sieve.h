/**
 * @brief Usual header file for prime sieve functions.
 *
 * @file sieve.h
 * @author your name
 * @date 2018-09-29
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
 * @param[out] sieve A boolean array indicating wether or not each index is prime or not. Initialized to `true`.
 * @param length The length of the given array.
 */
void prime_sieve( bool* sieve, size_t length );

/**
 * @brief Print the prime numbers from the given sieve.
 *
 * @param sieve
 * @param length
 */
void print_primes( const bool* sieve, size_t length );
