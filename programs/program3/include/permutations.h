#pragma once

#include "common.h"
#include <cstdint>
#include <cstddef>
#include <vector>

/**
 * @brief Get the @f$n@f$th permutation of numbers from 0 to the given length.
 *
 * @details See https://stackoverflow.com/a/7919887/3704977
 *
 * @param length The (exclusive) length.
 * @param n Which permutation to return.
 * @returns The @f$n@f$th permutation.
 */
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
