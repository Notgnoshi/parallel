#pragma once

#include "testsuite.h"

class CudaMultiplicationKernelTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( CudaMultiplicationKernelTest );
    CPPUNIT_TEST( SimpleSmall );
    CPPUNIT_TEST( MismatchedSmaller );
    CPPUNIT_TEST( MismatchedBigger );
    CPPUNIT_TEST( SimpleLarge );
    CPPUNIT_TEST( LargeMismatched );
    CPPUNIT_TEST_SUITE_END();

public:
    //! @brief Verify that multiplication works on simple small square matrices.
    void SimpleSmall();
    //! @brief Test with a matrix smaller than the block size.
    void MismatchedSmaller();
    //! @brief Test with a matrix bigger than, but not evenly divisible by, the block size.
    void MismatchedBigger();
    //! @brief Test with a large matrix that is evenly divisible by the block size.
    void SimpleLarge();
    //! @brief Test with a large matrix that is not evenly divisible by the block size.
    void LargeMismatched();
};

CPPUNIT_TEST_SUITE_REGISTRATION( CudaMultiplicationKernelTest );
