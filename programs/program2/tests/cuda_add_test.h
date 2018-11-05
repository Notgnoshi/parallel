#pragma once

#include "testsuite.h"

class CudaAdditionKernelTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( CudaAdditionKernelTest );
    CPPUNIT_TEST( SimpleAddition );
    CPPUNIT_TEST_SUITE_END();

public:
    //! @brief Verify that addition works on simple small matrices.
    void SimpleAddition();
};

CPPUNIT_TEST_SUITE_REGISTRATION( CudaAdditionKernelTest );
