#pragma once

#include "testsuite.h"

class KernelTest : public TestFixture
{
    CPPUNIT_TEST_SUITE( KernelTest );
    CPPUNIT_TEST( DefaultMultOperation );
    CPPUNIT_TEST( DefaultAddOperation );
    CPPUNIT_TEST_SUITE_END();

public:
    //! @brief Test the Kernel with the default multiplication kernel
    void DefaultMultOperation();
    //! @brief Test the Kernel with the default addition kernel
    void DefaultAddOperation();
};

CPPUNIT_TEST_SUITE_REGISTRATION( KernelTest );
