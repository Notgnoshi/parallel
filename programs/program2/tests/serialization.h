#pragma once

#include "matrix.h"
#include "testsuite.h"

class SerializationTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( SerializationTest );
    CPPUNIT_TEST( SimpleSmall );
    CPPUNIT_TEST_SUITE_END();

public:
    void SimpleSmall();
};

CPPUNIT_TEST_SUITE_REGISTRATION( SerializationTest );
