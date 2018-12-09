#include "test_serial.h"
#include "common.h"

void SerialStrategyTest::TestConstruction()
{
    //! @todo What the hell have I gotten myself into. This is going to suck for the parallel strategies.
    CPPUNIT_ASSERT( strategy.GetRank() == 0 );
}

void SerialStrategyTest::TestSmallSizes()
{
    for( size_t n = 2; n < 10; ++n )
    {
        CPPUNIT_ASSERT( strategy.Run( n ) == SOLUTIONS[n] );
    }
}

void SerialStrategyTest::TestClearVector()
{
    std::vector<bool> input( 4, true );
    const std::vector<bool> expected( 4, false );

    SerialStrategy::ClearVector( input );
    CPPUNIT_ASSERT( input == expected );
}

void SerialStrategyTest::TestIsSolution()
{
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

    CPPUNIT_ASSERT( this->strategy.IsSolution( {1, 3, 0, 2}, downhill, uphill ) );
    CPPUNIT_ASSERT( this->strategy.IsSolution( {2, 0, 3, 1}, downhill, uphill ) );

    for( auto&& perm : perms )
    {
        CPPUNIT_ASSERT( !this->strategy.IsSolution( perm, downhill, uphill ) );
    }
}
