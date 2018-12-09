#include "test_common.h"
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
