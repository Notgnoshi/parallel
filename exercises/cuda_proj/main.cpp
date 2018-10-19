#include "incr_wrapper.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>

/**
 * @brief Main entry point for the application.
 *
 * @returns 0 to indicate success.
 */
int main()
{
    double* a;
    double* b;
    size_t n = 100;

    a = (double*)malloc( n * sizeof( double ) );
    b = (double*)malloc( n * sizeof( double ) );

    for( size_t i = 0; i < n; ++i )
    {
        a[i] = 1;
        b[i] = 2;
    }

    incr( a, b, n );

    free( a );
    free( b );

    return 0;
}
