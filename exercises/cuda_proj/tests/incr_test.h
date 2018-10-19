#pragma once

#include "incr_wrapper.h"
#include "testsuite.h"

class IncrTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( IncrTest );
    CPPUNIT_TEST( SimpleIncr );
    CPPUNIT_TEST_SUITE_END();

public:
    void SimpleIncr();
};

CPPUNIT_TEST_SUITE_REGISTRATION( IncrTest );
