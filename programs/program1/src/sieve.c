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
    memset( a, true, length );

    return a;
}

void prime_sieve( bool* sieve, size_t length )
{
    // Only go up to the sqrt( length ).
    for( size_t i = 2; i < sqrt( ( double ) length ) + 1.5; ++i )
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
