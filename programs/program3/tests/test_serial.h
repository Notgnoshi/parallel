#pragma once

#include "strategies/serial.h"
#include "testsuite.h"

class SerialStrategyTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( SerialStrategyTest );
    CPPUNIT_TEST( TestConstruction );
    CPPUNIT_TEST( TestSmallSizes );
    CPPUNIT_TEST( TestClearVector );
    CPPUNIT_TEST( TestIsSolution );
    CPPUNIT_TEST_SUITE_END();

public:
    void TestConstruction();
    void TestSmallSizes();
    void TestClearVector();
    void TestIsSolution();

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
