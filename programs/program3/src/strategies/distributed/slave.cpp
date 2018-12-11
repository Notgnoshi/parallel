#include "strategies/distributed/slave.h"
#include "permutations.h"
#include <algorithm>
#include <vector>

size_t SlaveProcess::Run( bool screen_output, std::string file_output )
{
    size_t chunk_size = this->end_index - this->begin_index;
    size_t solutions = 0;
    std::vector<std::vector<uint8_t>> block;

    auto perm = NthPermutation( this->GetN(), this->begin_index );

    for( size_t i = 0; i < chunk_size; ++i )
    {
        if( this->IsSolution( perm ) )
        {
            ++solutions;
            if( screen_output && solutions < 10 )
            {
                PrintSolution( perm );
            }
            if( !file_output.empty() )
            {
                block.push_back( perm );

                if( block.size() >= 64 )
                {
                    AppendBlock( block, InsertRank( this->rank, file_output ) );
                    block.clear();
                }
            }
        }

        std::next_permutation( perm.begin(), perm.end() );
    }

    if( !file_output.empty() && block.size() > 0 )
    {
        AppendBlock( block, InsertRank( this->rank, file_output ) );
    }
    return solutions;
}

bool SlaveProcess::IsSolution( const std::vector<uint8_t>& arrangement )
{
    this->ClearDiagonals();

    for( uint8_t x = 0; x < this->GetN(); ++x )
    {
        auto up = static_cast<uint8_t>( x + arrangement[x] );
        auto down = static_cast<uint8_t>( x + n - arrangement[x] - 1 );

        if( this->downhill[down] || this->uphill[up] )
        {
            return false;
        }
        downhill[down] = true;
        uphill[up] = true;
    }

    return true;
}

std::string SlaveProcess::InsertRank( const size_t rank, const std::string& filename )
{
    size_t i = filename.find_last_of( '/' );
    std::string base = "";
    std::string name = filename;
    std::string ext = "";

    // Split off any path separators.
    if( i != std::string::npos )
    {
        base = filename.substr( 0, i );
        name = filename.substr( i );
    }

    i = name.find_last_of( '.' );
    // Split off any extensions.
    if( i != std::string::npos )
    {
        ext = name.substr( i );
        name = name.substr( 0, i );
    }

    return base + name + std::to_string( rank ) + ext;
}

void SlaveProcess::ClearDiagonals()
{
    for( size_t i = 0; i < this->uphill.size(); ++i )
    {
        this->uphill[i] = false;
        this->downhill[i] = false;
    }
}
