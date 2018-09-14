#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

int32_t main()
{
    size_t n = 10000000;
    clock_t tic, toc;

    // The usual reduction
    tic = clock();
    size_t sum1 = 0;
    #pragma omp parallel for num_threads( 8 ) reduction(+:sum1)
    for( size_t i = 0; i < n; ++i )
    {
        sum1 += i;
    }
    toc = clock();

    double elapsed = (double)(toc - tic) / CLOCKS_PER_SEC;
    printf( "reduced sum:  %zu\n", sum1 );
    printf( "elapsed time: %fs\n", elapsed );

    // A naive use of #pragma omp critical
    tic = clock();
    size_t sum2 = 0;
    size_t local_sum = 0;
    #pragma omp parallel for num_threads( 8 ) private( local_sum )
    for( size_t i = 0; i < n; ++i )
    {
        local_sum = i;

        // This is wrong. It's essentially sequential with a lot of overhead.
        #pragma omp critical
        sum2 += local_sum;
    }
    toc = clock();

    elapsed = (double)(toc - tic) / CLOCKS_PER_SEC;
    printf( "naive sum:    %zu\n", sum2 );
    printf( "elapsed time: %fs\n", elapsed );

    // The better way
    tic = clock();
    size_t sum3 = 0;
    #pragma omp parallel num_threads( 8 )
    {
        size_t local_sum = 0;
        #pragma omp for
        for(size_t i = 0; i < n; ++i )
        {
            local_sum += i;
        }

        #pragma omp critical
        sum3 += local_sum;
    }
    toc = clock();

    elapsed = (double)(toc - tic) / CLOCKS_PER_SEC;
    printf( "critical sum: %zu\n", sum3 );
    printf( "elapsed time: %fs\n", elapsed );

    return 0;
}
