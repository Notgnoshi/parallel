#include "permutations.h"

std::vector<uint8_t> NthPermutation( const size_t length, size_t n )
{
    std::vector<uint8_t> perm( length );
    for( size_t i = 0; i < length; ++i )
    {
        perm[i] = static_cast<uint8_t>( n / FACTORIALS[length - i - 1] );
        n %= FACTORIALS[length - i - 1];
    }

    for( size_t i = length - 1; i > 0; --i )
    {
        for( int64_t j = i - 1; j >= 0; --j )
        {
            if( perm[j] <= perm[i] )
            {
                perm[i]++;
            }
        }
    }

    return perm;
}
