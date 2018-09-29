/**
 * @brief Implementation of satisfiability functions. Yay.
 *
 * @file satisfiability.c
 * @author Austin Gill
 * @date 2018-09-24
 */
#include "satisfiability.h"

bool circuit_one( const int32_t tid, const uint16_t z )
{
    bool bits[16] = { 0 };
    extract_bits( bits, z );

    // For whatever reason, astyle indents the line continuation too far...
    if( ( bits[0]  ||  bits[1]  ) && ( !bits[1]  || !bits[3]  ) && ( bits[2]  ||  bits[3]  ) &&
            ( !bits[3]  || !bits[4]  ) && (  bits[4]  || !bits[5]  ) && ( bits[5]  || !bits[6]  ) &&
            (  bits[5]  ||  bits[6]  ) && (  bits[6]  || !bits[15] ) && ( bits[7]  || !bits[8]  ) &&
            ( !bits[7]  || !bits[13] ) && (  bits[8]  ||  bits[9]  ) && ( bits[8]  || !bits[9]  ) &&
            ( !bits[9]  || !bits[10] ) && (  bits[9]  ||  bits[11] ) && ( bits[10] ||  bits[11] ) &&
            (  bits[12] ||  bits[13] ) && (  bits[13] || !bits[14] ) && ( bits[14] ||  bits[15] ) )
    {
        printf( "%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", tid,
                bits[0], bits[1], bits[2],  bits[3],  bits[4],  bits[5],  bits[6],  bits[7],
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
    // I'd normally care a whole lot that there's tons of duplicated code here,
    // and I'd normally spend a good amount of time trying to come up with an elegant
    // solution, but I'm drunk, pissed off, and still in shock from my topology exam.
    // So this mess is what you get.
    #ifdef SCHEDULE_COMPARISON
        double begin = omp_get_wtime();
        size_t sum = 0;
        #pragma omp parallel for num_threads( omp_get_num_procs() )
        for( int32_t input = 0; input < USHRT_MAX; ++input )
        {
            if( circuit_fp( omp_get_thread_num(), (uint16_t) input ) )
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
        //! @note When using a `static` schedule, with chunk size of 1, the iteration variable must
        //! be signed. Or else you'll spend hours wondering why the simplest for loop you've ever
        //! written doesn't work... C.f. https://msdn.microsoft.com/en-us/library/b5b5b6eb.aspx
        #pragma omp parallel for num_threads( omp_get_num_procs() ) schedule( static, 1 )
        for( int32_t input = 0; input < USHRT_MAX; ++input )
        {
            if( circuit_fp( omp_get_thread_num(), (uint16_t) input ) )
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
        for( int32_t input = 0; input < USHRT_MAX; ++input )
        {
            if( circuit_fp( omp_get_thread_num(), (uint16_t) input ) )
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
        for( int32_t input = 0; input < USHRT_MAX; ++input )
        {
            if( circuit_fp( omp_get_thread_num(), (uint16_t) input ) )
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
