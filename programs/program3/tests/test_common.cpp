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

TEST_CASE( "SlaveProcess InsertRank" )
{
    const std::string filename1 = "test.txt";
    const size_t rank1 = 1;
    const std::string expected1 = "test1.txt";

    CHECK( expected1 == SlaveProcess::InsertRank( rank1, filename1 ) );

    const std::string filename2 = "test";
    const size_t rank2 = 2;
    const std::string expected2 = "test2";
    CHECK( expected2 == SlaveProcess::InsertRank( rank2, filename2 ) );

    const std::string filename3 = "../../test.sol";
    const size_t rank3 = 999;
    const std::string expected3 = "../../test999.sol";
    CHECK( expected3 == SlaveProcess::InsertRank( rank3, filename3 ) );

    const std::string filename4 = "/tmp/file_name.sol";
    const size_t rank4 = 0;
    const std::string expected4 = "/tmp/file_name0.sol";
    CHECK( expected4 == SlaveProcess::InsertRank( rank4, filename4 ) );

    const std::string filename5 = "../../test";
    const size_t rank5 = 3;
    const std::string expected5 = "../../test3";
    CHECK( expected5 == SlaveProcess::InsertRank( rank5, filename5 ) );
}
