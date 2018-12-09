#include "strategies/shared.h"
#include "common.h"
#include <algorithm>
#include <iomanip>

SharedStrategy::SharedStrategy( std::string file_output, bool screen_output, bool time ) :
    file_output( file_output ),
    screen_output( screen_output ),
    time( time )
{
}

size_t SharedStrategy::Run( size_t n )
{
    size_t solutions = 0;

    ThreadContext context( n, omp_get_num_procs() );

    Timer t;

// clang-format off
#pragma omp parallel for                                                      \
    default( none )                                                           \
    shared( n )                                                               \
    reduction( +: solutions )                                                 \
    firstprivate( context )                                                   \
    num_threads( omp_get_num_procs() )
    // clang-format on
    for( size_t i = 0; i < FACTORIALS[n]; ++i )
    {
        if( IsSolution( context.perm, context.downhill, context.uphill ) )
        {
            solutions += 1;
            // //! @todo This really isn't thread-safe.
            // //! @todo Determine how many solutions to print.
            // if( this->screen_output && solutions < 10 )
            // {
            //     PrintSolution( context.perm );
            // }
            // //! @todo This really isn't thread-safe.
            // //! @todo Determine the right chunk size.
            // if( !this->file_output.empty() && chunk.size() < 64 )
            // {
            //     chunk.push_back( context.perm );

            //     if( chunk.size() == 64 )
            //     {
            //         AppendBlock( chunk, this->file_output );
            //         chunk.clear();
            //     }
            // }
        }

        std::next_permutation( context.perm.begin(), context.perm.end() );
    }

    if( this->time )
    {
        std::cout << std::fixed << std::setprecision( 6 );
        std::cout << "Elapsed time: " << t.elapsed() << "s" << std::endl;
    }

    // // If there are leftover solutions, print them.
    // if( !this->file_output.empty() && chunk.size() > 0 )
    // {
    //     AppendBlock( chunk, this->file_output );
    // }
    return solutions;
}

bool SharedStrategy::IsSolution( const std::vector<uint8_t>& arrangement, std::vector<bool>& downhill, std::vector<bool>& uphill )
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

void SharedStrategy::ClearVector( std::vector<bool>& v )
{
    // Can't use memset because the bool specialization does not necessarily store
    // the elements in contiguous arrays.
    for( auto&& i : v )
    {
        i = false;
    }
}
