#include "permutations.h"

std::vector<uint8_t> NthPermutation( size_t length, size_t n )
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

bool IsSolution( std::vector<uint8_t> arrangement )
{
    auto n = static_cast<uint8_t>( arrangement.size() );
    std::vector<bool> downhill( 2 * n - 1, false );
    std::vector<bool> uphill( 2 * n - 1, false );

    // Attempt to place the queen in the diagonal arrays. If we can't, return false.
    for( uint8_t x = 0; x < n; ++x )
    {
        auto up = static_cast<uint8_t>( x + arrangement[x] );
        auto down = static_cast<uint8_t>( x + n - arrangement[x] - 1 );

        if( downhill[down] || uphill[up] )
        {
            return false;
        }
        downhill[down] = true;
        uphill[up] = true;
    }

    return true;
}
