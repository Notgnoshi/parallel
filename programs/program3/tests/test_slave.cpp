#include "catch.hpp"
#include "strategies/distributed/slave.h"

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

TEST_CASE( "SlaveProcess IsSolution" )
{
    // All of the solutions for n = 5
    const std::vector<std::vector<uint8_t>> solutions = {
        {0, 2, 4, 1, 3},
        {0, 3, 1, 4, 2},
        {1, 3, 0, 2, 4},
        {1, 4, 2, 0, 3},
        {2, 0, 3, 1, 4},
        {2, 4, 1, 3, 0},
        {3, 0, 2, 4, 1},
        {3, 1, 4, 2, 0},
        {4, 1, 3, 0, 2},
        {4, 2, 0, 3, 1},
    };

    // Divide the permutations into 4 groups (plus the master).
    auto slave = SlaveProcess( 1, 5, 5 );

    size_t solution = 0;
    std::vector<uint8_t> perm = {0, 1, 2, 3, 4};
    for( size_t i = 0; i < FACTORIALS[5]; ++i )
    {
        // Make sure we don't find any false positives.
        if( slave.IsSolution( perm ) )
        {
            CHECK( perm == solutions[solution] );
            ++solution;
        }

        std::next_permutation( perm.begin(), perm.end() );
    }

    // Make sure all of the known solutions are marked as solutions.
    for( auto&& sol : solutions )
    {
        CHECK( slave.IsSolution( sol ) );
    }
}
