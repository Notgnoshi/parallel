#pragma once

#include "testsuite.h"

class CpuMultKernelTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( CpuMultKernelTest );
    CPPUNIT_TEST( MatVectMult );
    CPPUNIT_TEST( LessSimple );
    CPPUNIT_TEST_SUITE_END();

public:
    //! @brief Verify that the multiplication works.
    void MatVectMult();
    //! @brief ...
    void LessSimple();
};

CPPUNIT_TEST_SUITE_REGISTRATION( CpuMultKernelTest );
