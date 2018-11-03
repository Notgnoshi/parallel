#pragma once

#include "kernels/cpu_mult.h"
#include "testsuite.h"

class CpuMultKernelTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( CpuMultKernelTest );
    CPPUNIT_TEST( SimpleDotProduct );
    CPPUNIT_TEST( MatrixDotProduct );
    CPPUNIT_TEST( MatVectMult );
    CPPUNIT_TEST_SUITE_END();

public:
    //! @brief Test that the dot product works on less complicated arrays.
    void SimpleDotProduct();
    //! @brief Verify that the dot product works on Matrix_t structs.
    void MatrixDotProduct();
    //! @brief Verify that the multiplication works.
    void MatVectMult();
};

CPPUNIT_TEST_SUITE_REGISTRATION( CpuMultKernelTest );
