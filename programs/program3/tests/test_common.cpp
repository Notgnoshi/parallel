#include "catch.hpp"
#include "common.h"
#include "permutations.h"
#include "strategies/distributed/slave.h"
#include <algorithm>

TEST_CASE( "Common NthPermutation" )
{
    const std::vector<std::vector<uint8_t>> expected = {
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
        {1, 3, 0, 2},
        {1, 3, 2, 0},
        {2, 0, 1, 3},
        {2, 0, 3, 1},
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

    for( size_t i = 0; i < expected.size(); ++i )
    {
        CHECK( expected[i] == NthPermutation( 4, i ) );
    }

    // Make sure next_permutation works like we expect it to.
    for( size_t i = 0; i < expected.size() - 1; ++i )
    {
        std::vector<uint8_t> elem = expected[i];
        std::next_permutation( elem.begin(), elem.end() );
        CHECK( elem == NthPermutation( 4, i + 1 ) );
    }
}

TEST_CASE( "Common IsSolution" )
{
    const std::vector<uint8_t> solution = {2, 0, 3, 1};
    const std::vector<uint8_t> failure1 = {1, 0, 3, 2};
    const std::vector<uint8_t> failure2 = {2, 3, 0, 1};
    const std::vector<uint8_t> failure3 = {0, 1, 2, 3};

    CHECK( IsSolution( solution ) );
    CHECK( !IsSolution( failure1 ) );
    CHECK( !IsSolution( failure2 ) );
    CHECK( !IsSolution( failure3 ) );
}

TEST_CASE( "Common IsSolution for n=4" )
{
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

    CHECK( IsSolution( {1, 3, 0, 2} ) );
    CHECK( IsSolution( {2, 0, 3, 1} ) );

    for( auto&& perm : perms )
    {
        CHECK( !IsSolution( perm ) );
    }
}
