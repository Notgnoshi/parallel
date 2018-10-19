#include "incr_test.h"

void IncrTest::SimpleIncr()
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

    for( size_t i = 0; i < n; ++i )
    {
        CPPUNIT_ASSERT( a[i] == 3 );
    }

    free( a );
    free( b );
}
