/**
 * @file pmain.c
 * @author Austin Gill (atgill@protonmail.com)
 * @brief The main entry point and commandline argument handling portion of the program.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "sieve.h"

/**
 * @brief Print the program's usage statement.
 */
void usage()
{
    printf( "Computes the prime sieve for numbers less than the given limit.\n" );
    printf( "Usage: ./prime <limit>\n" );
}

/**
 * @brief Get limit for sieve from commandline arguments.
 *
 * Prints usage statement and exits program on failure.
 *
 * @param argc The usual.
 * @param argv The usual.
 */
size_t parse_args( int argc, char const** argv )
{
    if( argc == 2 )
    {
        size_t limit = 0;
        sscanf( argv[1], "%zu", &limit );
        return limit;
    }
    else
    {
        usage();
        exit( 1 );
    }
}

/**
 * @brief The main entry point for the prime sieve program.
 *
 * Uses the Sieve of Eratosthenes to generate the first @f$n@f$ prime numbers.
 *
 * Base on compile-time `-D` defines, we will also time the prime sieve to
 * compare runtimes.
 *
 * @param argc The usual number of commandline arguments.
 * @param argv The usual commandline arguments.
 */
int main( int argc, char const** argv )
{
    size_t limit = parse_args( argc, argv );

    // Length of sieve is limit + 1 because of 0 indexing.
    bool* sieve = allocate( limit + 1 );

    // Run the prime sieve
    double elapsed = prime_sieve( sieve, limit + 1 );

    // Print the primes
    print_primes( sieve, limit + 1 );
    printf( "Elapsed time: %f ms\n", elapsed );

    free( sieve );
    return 0;
}
