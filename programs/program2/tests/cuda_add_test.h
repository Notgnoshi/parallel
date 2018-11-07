#pragma once

#include "testsuite.h"

class CudaAdditionKernelTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( CudaAdditionKernelTest );
    CPPUNIT_TEST( SimpleAddition );
    CPPUNIT_TEST( LargeAddition );
    CPPUNIT_TEST( MismatchedLarger );
    CPPUNIT_TEST( MismatchedSmaller );
    CPPUNIT_TEST( LargeMismatched );
    CPPUNIT_TEST_SUITE_END();

public:
    //! @brief Verify that addition works on simple small matrices.
    void SimpleAddition();
    //! @brief Test addition with a matrix that's evenly divisible by several blocks.
    void LargeAddition();
    //! @brief Test addition with matrix sizes bigger, but not evenly divisible by the block size.
    void MismatchedLarger();
    //! @brief Test addition with matrix sizes smaller than the block size.
    void MismatchedSmaller();
    //! @brief Test addition with a large matrix that is not evenly divisible.
    void LargeMismatched();
};

CPPUNIT_TEST_SUITE_REGISTRATION( CudaAdditionKernelTest );
