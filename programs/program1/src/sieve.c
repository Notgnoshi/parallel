/**
 * @file sieve.c
 * @author Austin Gill (atgill@protonmail.com)
 * @brief Implementation of the Sieve of Eratosthenes.
 */
#include "sieve.h"

bool* allocate( size_t length )
{
    bool* a = malloc( length * sizeof( bool ) );

    if( a == NULL )
    {
        fprintf( stderr, "Failed to allocate %zu bytes\n", length * sizeof( bool ) );
        exit( 1 );
    }
    // Initialize the sieve with every number marked as prime.
    memset( a, true, length );

    return a;
}

double prime_sieve( bool* sieve, size_t length )
{
    double begin = omp_get_wtime();
// Use a dynamic schedule because some iterations will take longer than others.
#pragma omp parallel for num_threads( omp_get_num_procs() ) schedule( dynamic )
    for( size_t i = 2; i < (size_t)sqrt( (double)length ); ++i )
    {
        if( sieve[i] )
        {
            // Mark all multiples of i as composite
            for( size_t j = i * i; j < length; j += i )
            {
                sieve[j] = false;
            }
        }
    }
    double end = omp_get_wtime();
    // Convert seconds to milliseconds
    return ( end - begin ) * 1000;
}

void print_primes( const bool* sieve, size_t length )
{
    size_t primes = 0;
    for( size_t i = 0; i < length; ++i )
    {
        if( sieve[i] )
        {
            if( primes % 10 == 0 )
            {
                printf( "\n%zu:\t", primes );
            }
            ++primes;
            printf( "%zu\t", i );
        }
    }
    printf( "\n" );
}
