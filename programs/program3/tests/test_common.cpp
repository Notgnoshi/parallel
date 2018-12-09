#include "test_common.h"
#include "common.h"
#include "permutations.h"
#include <algorithm>

void CommonTest::TestNthPermutation()
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
        CPPUNIT_ASSERT( expected[i] == NthPermutation( 4, i ) );
    }

    // Make sure next_permutation works like we expect it to.
    for( size_t i = 0; i < expected.size() - 1; ++i )
    {
        std::vector<uint8_t> elem = expected[i];
        std::next_permutation( elem.begin(), elem.end() );
        CPPUNIT_ASSERT( elem == NthPermutation( 4, i + 1 ) );
    }
}

void CommonTest::TestIsSolution()
{
    const std::vector<uint8_t> solution = {2, 0, 3, 1};
    const std::vector<uint8_t> failure1 = {1, 0, 3, 2};
    const std::vector<uint8_t> failure2 = {2, 3, 0, 1};
    const std::vector<uint8_t> failure3 = {0, 1, 2, 3};

    CPPUNIT_ASSERT( IsSolution( solution ) );
    CPPUNIT_ASSERT( !IsSolution( failure1 ) );
    CPPUNIT_ASSERT( !IsSolution( failure2 ) );
    CPPUNIT_ASSERT( !IsSolution( failure3 ) );
}

void CommonTest::TestIsSolution4()
{
    const size_t n = 4;
    size_t sol_index = 0;
    // Ordering takes advantage of NthPermutation being ordered.
    std::vector<std::vector<uint8_t>> solutions = {
        {1, 3, 0, 2},
        {2, 0, 3, 1},
    };

    for( size_t i = 0; i < FACTORIALS[n]; ++i )
    {
        std::vector<uint8_t> perm = NthPermutation( 4, i );
        if( IsSolution( perm ) )
        {
            CPPUNIT_ASSERT( perm == solutions[sol_index] );
            ++sol_index;
        }
    }

    CPPUNIT_ASSERT( sol_index == 2 );
}
