#include "satisfiability.h"

bool circuit_one( const uint32_t pid, const uint16_t z )
{
    //! @todo Implement.
    return pid + z;
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
