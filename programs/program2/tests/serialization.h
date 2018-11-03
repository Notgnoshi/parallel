#pragma once

#include "matrix.h"
#include "testsuite.h"

class SerializationTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( SerializationTest );
    CPPUNIT_TEST( SimpleSmall );
    CPPUNIT_TEST( SimplePack );
    CPPUNIT_TEST_SUITE_END();

public:
    //! @brief Verify that serialization and deserialization works.
    void SimpleSmall();
    //! @brief Verify that the python `pack.py` script works.
    void SimplePack();
};

CPPUNIT_TEST_SUITE_REGISTRATION( SerializationTest );
