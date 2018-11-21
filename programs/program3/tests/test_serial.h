#pragma once

#include "strategies/serial.h"
#include "testsuite.h"

class SerialStrategyTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( SerialStrategyTest );
    CPPUNIT_TEST( TestConstruction );
    CPPUNIT_TEST( TestSmallSizes );
    CPPUNIT_TEST_SUITE_END();

public:
    void TestConstruction();
    void TestSmallSizes();

    void setUp()
    {
        this->strategy = SerialStrategy( "", false );
    }

    void tearDown()
    {
    }

private:
    SerialStrategy strategy;
};

CPPUNIT_TEST_SUITE_REGISTRATION( SerialStrategyTest );
