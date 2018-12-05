#pragma once

#include "strategies/serial.h"
#include "testsuite.h"

class CommonTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( CommonTest );
    CPPUNIT_TEST( TestNthPermutation );
    CPPUNIT_TEST_SUITE_END();

public:
    void TestNthPermutation();

private:
};

CPPUNIT_TEST_SUITE_REGISTRATION( CommonTest );
