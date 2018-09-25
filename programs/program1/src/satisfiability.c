#include "satisfiability.h"

bool circuit_one( const uint32_t pid, const uint16_t z )
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
        printf( "%d) %d%d%d%d%d%d%d%d%d%d%d%d%d%d%d%d\n", pid,
                bits[0], bits[1], bits[2], bits[3], bits[4], bits[5], bits[6], bits[7],
                bits[8], bits[9], bits[10], bits[11], bits[12], bits[13], bits[14], bits[15] );
        fflush ( stdout );
        return true;
    }

    return false;
}

bool circuit_two( const uint32_t pid, const uint16_t z )
{
    //! @todo Implement.
    return pid + z;
}

static inline void extract_bits( bool* bits, const uint16_t z )
{
    for( size_t i = 0; i < 16; ++i )
    {
        bits[i] = EXTRACT_BIT( z, i );
    }
}
