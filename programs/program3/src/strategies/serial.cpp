#include "strategies/serial.h"
#include "common.h"
#include "permutations.h"
#include <algorithm>

SerialStrategy::SerialStrategy( std::string file_output, bool time ) :
    file_output( file_output ),
    time( time )
{
}

size_t SerialStrategy::Run( size_t n, bool screen_output )
{
    size_t solutions = 0;
    bool file_output = !this->file_output.empty();
    std::vector<std::vector<uint8_t>> chunk;
    std::vector<uint8_t> perm = NthPermutation( n, 0 );

    std::vector<bool> downhill( 2 * n - 1, false );
    std::vector<bool> uphill( 2 * n - 1, false );

    for( size_t i = 1; i < FACTORIALS[n]; ++i )
    {
        if( this->IsSolution( perm, downhill, uphill ) )
        {
            ++solutions;
            //! @todo Determine how many solutions to print.
            if( screen_output && solutions < 10 )
            {
                PrintSolution( perm );
            }

            //! @todo Determine the right chunk size.
            if( file_output && chunk.size() < 64 )
            {
                chunk.push_back( perm );

                if( chunk.size() == 64 )
                {
                    AppendBlock( chunk, this->file_output );
                    chunk.clear();
                }
            }
        }

        // Force an early exit.
        if( solutions == SOLUTIONS[n] )
        {
            return solutions;
        }

        std::next_permutation( perm.begin(), perm.end() );
    }
    return solutions;
}

bool SerialStrategy::IsSolution( const std::vector<uint8_t>& arrangement, std::vector<bool>& downhill, std::vector<bool>& uphill )
{
    const uint8_t n = static_cast<uint8_t>( arrangement.size() );
    ClearVector( downhill );
    ClearVector( uphill );

    // Attempt to place each queen in the diagonal arrays. If we can't, return false.
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

void SerialStrategy::ClearVector( std::vector<bool>& v )
{
    // Can't use memset because the bool specialization does not necessarily store
    // the elements in contiguous arrays.
    for( auto&& i : v )
    {
        i = false;
    }
}
