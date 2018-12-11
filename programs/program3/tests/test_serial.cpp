#include "catch.hpp"
#include "common.h"
#include "strategies/serial.h"

TEST_CASE( "SerialStrategy construction" )
{
    SerialStrategy strategy;
    CHECK( strategy.GetRank() == 0 );
}

TEST_CASE( "SerialStrategy Run" )
{
    SerialStrategy strategy;
    for( size_t n = 2; n < 10; ++n )
    {
        CHECK( strategy.Run( n ) == SOLUTIONS[n] );
    }
}

TEST_CASE( "SerialStrategy ClearVector" )
{
    std::vector<bool> input( 4, true );
    const std::vector<bool> expected( 4, false );

    SerialStrategy::ClearVector( input );
    CHECK( input == expected );
}

TEST_CASE( "SerialStrategy IsSolution" )
{
    SerialStrategy strategy;
    const size_t n = 4;
    const std::vector<std::vector<uint8_t>> perms = {
        {0, 1, 2, 3},
        {0, 1, 3, 2},
        {0, 2, 1, 3},
        {0, 2, 3, 1},
        {0, 3, 1, 2},
        {0, 3, 2, 1},
        {1, 0, 2, 3},
        {1, 0, 3, 2},
        {1, 2, 0, 3},
        {1, 2, 3, 0},
        // {1, 3, 0, 2},
        {1, 3, 2, 0},
        {2, 0, 1, 3},
        // {2, 0, 3, 1},
        {2, 1, 0, 3},
        {2, 1, 3, 0},
        {2, 3, 0, 1},
        {2, 3, 1, 0},
        {3, 0, 1, 2},
        {3, 0, 2, 1},
        {3, 1, 0, 2},
        {3, 1, 2, 0},
        {3, 2, 0, 1},
        {3, 2, 1, 0},
    };

    std::vector<bool> downhill( 2 * n - 1, false );
    std::vector<bool> uphill( 2 * n - 1, false );

    CHECK( strategy.IsSolution( {1, 3, 0, 2}, downhill, uphill ) );
    CHECK( strategy.IsSolution( {2, 0, 3, 1}, downhill, uphill ) );

    for( auto&& perm : perms )
    {
        CHECK( !strategy.IsSolution( perm, downhill, uphill ) );
    }
}
