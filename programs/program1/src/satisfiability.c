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
    //! @todo Add timing code with `ifdef`s.
    //! @todo Use `ifdef`s for schedule.
    //! @todo Add parallelization code
    //! @todo What does static/dynamic schedule mean?
    //! @todo Copy/paste code to do static/dynamic.
    size_t sum = 0;
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
}
