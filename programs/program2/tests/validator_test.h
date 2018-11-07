#pragma once

#include "testsuite.h"

class ValidationTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( ValidationTest );
    CPPUNIT_TEST( ValidateAddition );
    CPPUNIT_TEST( ValidateMultiplication );
    CPPUNIT_TEST( ValidateMultiplicationKernels );
    CPPUNIT_TEST( ValidateAdditionKernels );
    CPPUNIT_TEST_SUITE_END();

public:
    //! @brief Verify that the addition validator works.
    void ValidateAddition();
    //! @brief Verify that the multiplication validator works.
    void ValidateMultiplication();
    //! @brief Verify that the multiplication kernels handle validation correctly.
    void ValidateMultiplicationKernels();
    //! @brief Verify that the addition kernels handle validation correctly.
    void ValidateAdditionKernels();
};

CPPUNIT_TEST_SUITE_REGISTRATION( ValidationTest );
