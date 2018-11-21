#include "test_serial.h"
#include "common.h"

void SerialStrategyTest::TestConstruction()
{
    //! @todo What the hell have I gotten myself into. This is going to suck for the parallel strategies.
    CPPUNIT_ASSERT( strategy.GetRank() == 0 );
}

void SerialStrategyTest::TestSmallSizes()
{
    for( size_t n = 0; n < 8; ++n )
    {
        CPPUNIT_ASSERT( strategy.Run( n, false ) == SOLUTIONS[n] );
    }
}
