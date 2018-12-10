#include "catch.hpp"
#include "strategies/distributed/master.h"
#include "strategies/distributed/process_factory.h"
#include "strategies/distributed/slave.h"

TEST_CASE( "Process work division" )
{
    SECTION( "Verify construction with evenly divisible work" )
    {
        // There are 9 total processes; one master and eight workers.
        auto master = MasterProcess( 0, 9, 4 );
        auto slave1 = SlaveProcess( 1, 9, 4 );
        auto slave2 = SlaveProcess( 2, 9, 4 );

        // The master is responsible for no work.
        CHECK( master.GetBegin() == 0 );
        CHECK( master.GetEnd() == 0 );

        // There are 8 slaves, so each slave is responsible for 3 units of work.
        CHECK( slave1.GetBegin() == 0 );
        // The end points are exclusive.
        CHECK( slave1.GetEnd() == 3 );

        CHECK( slave2.GetBegin() == 3 );
        CHECK( slave2.GetEnd() == 6 );

        // It's always a good idea to check the end workers.
        auto slave8 = SlaveProcess( 8, 9, 4 );
        CHECK( slave8.GetBegin() == 21 );
        CHECK( slave8.GetEnd() == 24 );
    }

    SECTION( "Verify construction using too many processes" )
    {
        auto master = GetProcess( 0, 10, 4 );
        auto slave1 = GetProcess( 1, 10, 4 );
        auto slave2 = GetProcess( 2, 10, 4 );
        auto slave8 = GetProcess( 8, 10, 4 );
        auto slave9 = GetProcess( 9, 10, 4 );

        CHECK( master->GetBegin() == 0 );
        CHECK( master->GetEnd() == 0 );

        // The number of processes doesn't evenly divide the amount of work,
        // Make each process round up so the last worker doesn't have to do
        // twice the work.
        CHECK( slave1->GetBegin() == 0 );
        // The chunk size rounds up to three.
        CHECK( slave1->GetEnd() == 3 );

        CHECK( slave2->GetBegin() == 3 );
        CHECK( slave2->GetEnd() == 6 );

        CHECK( slave8->GetBegin() == 21 );
        CHECK( slave8->GetEnd() == 24 );

        // This leaves the ninth slave with nothing to do.
        CHECK( slave9->GetBegin() == 24 );
        CHECK( slave9->GetEnd() == 24 );
    }

    SECTION( "Verify construction using too few processes" )
    {
        auto master = GetProcess( 0, 8, 4 );
        auto slave1 = GetProcess( 1, 8, 4 );
        auto slave2 = GetProcess( 2, 8, 4 );
        auto slave6 = GetProcess( 6, 8, 4 );
        auto slave7 = GetProcess( 7, 8, 4 );

        CHECK( master->GetBegin() == 0 );
        CHECK( master->GetEnd() == 0 );

        CHECK( slave1->GetBegin() == 0 );
        // The chunk size rounds up to four.
        CHECK( slave1->GetEnd() == 4 );

        CHECK( slave2->GetBegin() == 4 );
        CHECK( slave2->GetEnd() == 8 );

        CHECK( slave6->GetBegin() == 20 );
        CHECK( slave6->GetEnd() == 24 );

        CHECK( slave7->GetBegin() == 24 );
        CHECK( slave7->GetEnd() == 24 );
    }

    SECTION( "Verify construction when there are as many processes as tasks" )
    {
        auto master = GetProcess( 0, 7, 3 );
        auto slave1 = GetProcess( 1, 7, 3 );
        auto slave2 = GetProcess( 2, 7, 3 );
        auto slave3 = GetProcess( 3, 7, 3 );
        auto slave4 = GetProcess( 4, 7, 3 );
        auto slave5 = GetProcess( 5, 7, 3 );
        auto slave6 = GetProcess( 6, 7, 3 );

        CHECK( master->GetBegin() == 0 );
        CHECK( master->GetEnd() == 0 );

        CHECK( slave1->GetBegin() == 0 );
        CHECK( slave1->GetEnd() == 1 );
        CHECK( slave2->GetBegin() == 1 );
        CHECK( slave2->GetEnd() == 2 );
        CHECK( slave3->GetBegin() == 2 );
        CHECK( slave3->GetEnd() == 3 );
        CHECK( slave4->GetBegin() == 3 );
        CHECK( slave4->GetEnd() == 4 );
        CHECK( slave5->GetBegin() == 4 );
        CHECK( slave5->GetEnd() == 5 );
        CHECK( slave6->GetBegin() == 5 );
        CHECK( slave6->GetEnd() == 6 );
    }

    SECTION( "Verify construction when there are more processes than tasks" )
    {
        auto slave1 = GetProcess( 1, 5, 2 );
        auto slave2 = GetProcess( 2, 5, 2 );
        auto slave3 = GetProcess( 3, 5, 2 );
        auto slave4 = GetProcess( 4, 5, 2 );

        CHECK( slave1->GetBegin() == 0 );
        CHECK( slave1->GetEnd() == 1 );

        CHECK( slave2->GetBegin() == 1 );
        CHECK( slave2->GetEnd() == 2 );

        // The last worker(s) get assigned no work.
        CHECK( slave3->GetBegin() == 2 );
        CHECK( slave3->GetEnd() == 2 );
        CHECK( slave4->GetBegin() == 2 );
        CHECK( slave4->GetEnd() == 2 );
    }
}
