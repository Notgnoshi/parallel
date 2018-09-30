/**
 * @file satisfiability.c
 * @author Austin Gill (atgill@protonmail.com)
 * @brief Implementation of satisfiability functions. Yay.
 */
#include "satisfiability.h"

bool circuit_one( const int32_t tid, const uint16_t z )
{
    bool bits[16] = {0};
    extract_bits( bits, z );

    if( ( bits[0] || bits[1] ) && ( !bits[1] || !bits[3] ) && ( bits[2] || bits[3] ) &&
        ( !bits[3] || !bits[4] ) && ( bits[4] || !bits[5] ) && ( bits[5] || !bits[6] ) &&
        ( bits[5] || bits[6] ) && ( bits[6] || !bits[15] ) && ( bits[7] || !bits[8] ) &&
        ( !bits[7] || !bits[13] ) && ( bits[8] || bits[9] ) && ( bits[8] || !bits[9] ) &&
        ( !bits[9] || !bits[10] ) && ( bits[9] || bits[11] ) && ( bits[10] || bits[11] ) &&
        ( bits[12] || bits[13] ) && ( bits[13] || !bits[14] ) && ( bits[14] || bits[15] ) )
    {
        printf( "%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", tid,
                bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7],
                bits[8], bits[9], bits[10], bits[11], bits[12], bits[13], bits[14], bits[15] );
        fflush( stdout );
        return true;
    }

    return false;
}

bool circuit_two( const int32_t tid, const uint16_t z )
{
    //! @todo Create another circuit with a small number of satisfied inputs.
    return tid + z;
}

void check_circuit( bool ( *circuit_fp )( const int32_t, const uint16_t ) )
{
#ifdef SCHEDULE_COMPARISON
    double begin = omp_get_wtime();
    size_t sum = 0;
#pragma omp parallel for num_threads( omp_get_num_procs() )
    for( uint16_t input = 0; input < USHRT_MAX; ++input )
    {
        if( circuit_fp( omp_get_thread_num(), input ) )
        {
            ++sum;
        }
    }
    double end = omp_get_wtime();

    printf( "\n" );
    printf( "=============================================================================\n" );
    printf( "found %zu inputs satisfying the circuit in %4f seconds with default schedule.\n",
            sum, end - begin );
    printf( "=============================================================================\n" );
    printf( "\n" );

    sum = 0;
    begin = omp_get_wtime();
#pragma omp parallel for num_threads( omp_get_num_procs() ) schedule( static, 1 )
    for( uint32_t input = 0; input < USHRT_MAX; ++input )
    {
        if( circuit_fp( omp_get_thread_num(), (uint16_t)input ) )
        {
            ++sum;
        }
    }
    end = omp_get_wtime();

    printf( "\n" );
    printf( "=============================================================================\n" );
    printf( "found %zu inputs satisfying the circuit in %4f seconds with static schedule.\n",
            sum, end - begin );
    printf( "=============================================================================\n" );
    printf( "\n" );

    sum = 0;
    begin = omp_get_wtime();
#pragma omp parallel for num_threads( omp_get_num_procs() ) schedule( dynamic, 1 )
    for( uint16_t input = 0; input < USHRT_MAX; ++input )
    {
        if( circuit_fp( omp_get_thread_num(), input ) )
        {
            ++sum;
        }
    }
    end = omp_get_wtime();

    printf( "\n" );
    printf( "=============================================================================\n" );
    printf( "found %zu inputs satisfying the circuit in %4f seconds with dynamic schedule.\n",
            sum, end - begin );
    printf( "=============================================================================\n" );
#else
    size_t sum = 0;
#pragma omp parallel for num_threads( omp_get_num_procs() )
    for( uint16_t input = 0; input < USHRT_MAX; ++input )
    {
        if( circuit_fp( omp_get_thread_num(), input ) )
        {
            ++sum;
        }
    }

    printf( "\n" );
    printf( "=======================================\n" );
    printf( "%zu inputs satisfied the given circuit.\n", sum );
    printf( "=======================================\n" );
#endif // SCHEDULE_COMPARISON
}
