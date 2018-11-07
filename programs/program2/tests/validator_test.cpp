#include "validator_test.h"
#include "kernels/kernel.h"
#include "kernels/kernel_factory.h"
#include "matrix.h"
#include "validator.h"

void ValidationTest::ValidateAddition()
{
    CPPUNIT_ASSERT( AdditionValidator( Matrix_t( 0, 0 ), Matrix_t( 0, 0 ) ) );
    CPPUNIT_ASSERT( AdditionValidator( Matrix_t( 1, 1 ), Matrix_t( 1, 1 ) ) );
    CPPUNIT_ASSERT( AdditionValidator( Matrix_t( 16, 16 ), Matrix_t( 16, 16 ) ) );
    CPPUNIT_ASSERT( AdditionValidator( Matrix_t( 234, 1234 ), Matrix_t( 234, 1234 ) ) );
    CPPUNIT_ASSERT( AdditionValidator( Matrix_t( 13, 1 ), Matrix_t( 13, 1 ) ) );

    CPPUNIT_ASSERT( !AdditionValidator( Matrix_t( 15, 16 ), Matrix_t( 16, 16 ) ) );
    CPPUNIT_ASSERT( !AdditionValidator( Matrix_t( 16, 15 ), Matrix_t( 16, 16 ) ) );
    CPPUNIT_ASSERT( !AdditionValidator( Matrix_t( 16, 16 ), Matrix_t( 15, 16 ) ) );
    CPPUNIT_ASSERT( !AdditionValidator( Matrix_t( 16, 16 ), Matrix_t( 16, 15 ) ) );
}

void ValidationTest::ValidateMultiplication()
{
    CPPUNIT_ASSERT( MultiplicationValidator( Matrix_t( 4, 4 ), Matrix_t( 4, 4 ) ) );
    CPPUNIT_ASSERT( MultiplicationValidator( Matrix_t( 0, 0 ), Matrix_t( 0, 0 ) ) );
    CPPUNIT_ASSERT( MultiplicationValidator( Matrix_t( 1, 1 ), Matrix_t( 1, 1 ) ) );

    CPPUNIT_ASSERT( MultiplicationValidator( Matrix_t( 4, 4 ), Matrix_t( 4, 3 ) ) );
    CPPUNIT_ASSERT( MultiplicationValidator( Matrix_t( 4, 4 ), Matrix_t( 4, 5 ) ) );
    CPPUNIT_ASSERT( MultiplicationValidator( Matrix_t( 4, 4 ), Matrix_t( 4, 1 ) ) );

    CPPUNIT_ASSERT( MultiplicationValidator( Matrix_t( 1, 4 ), Matrix_t( 4, 4 ) ) );
    CPPUNIT_ASSERT( MultiplicationValidator( Matrix_t( 3, 4 ), Matrix_t( 4, 3 ) ) );
    CPPUNIT_ASSERT( MultiplicationValidator( Matrix_t( 5, 4 ), Matrix_t( 4, 5 ) ) );

    CPPUNIT_ASSERT( !MultiplicationValidator( Matrix_t( 4, 3 ), Matrix_t( 4, 4 ) ) );
    CPPUNIT_ASSERT( !MultiplicationValidator( Matrix_t( 4, 5 ), Matrix_t( 4, 4 ) ) );
    CPPUNIT_ASSERT( !MultiplicationValidator( Matrix_t( 4, 4 ), Matrix_t( 3, 4 ) ) );
    CPPUNIT_ASSERT( !MultiplicationValidator( Matrix_t( 4, 4 ), Matrix_t( 5, 4 ) ) );

    CPPUNIT_ASSERT( !MultiplicationValidator( Matrix_t( 4, 3 ), Matrix_t( 4, 3 ) ) );
    CPPUNIT_ASSERT( !MultiplicationValidator( Matrix_t( 4, 5 ), Matrix_t( 4, 5 ) ) );

    CPPUNIT_ASSERT( !MultiplicationValidator( Matrix_t( 1, 4 ), Matrix_t( 3, 4 ) ) );
    CPPUNIT_ASSERT( !MultiplicationValidator( Matrix_t( 3, 4 ), Matrix_t( 5, 3 ) ) );
}

void ValidationTest::ValidateMultiplicationKernels()
{
    const Matrix_t expected( 0, 0 );
    auto kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_DEFAULT ).GetKernel();

    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 3 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 3, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 5 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 5, 4 ) ) == expected );

    kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CPU ).GetKernel();

    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 3 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 3, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 5 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 5, 4 ) ) == expected );

    kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CUDA ).GetKernel();

    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 3 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 3, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 5 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 5, 4 ) ) == expected );
}

void ValidationTest::ValidateAdditionKernels()
{
    const Matrix_t expected( 0, 0 );
    auto kernel = KernelFactory( OPERATION_ADDITION, KERNEL_DEFAULT ).GetKernel();

    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 3 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 3, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 5 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 5, 4 ) ) == expected );

    kernel = KernelFactory( OPERATION_ADDITION, KERNEL_CPU ).GetKernel();

    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 3 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 3, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 5 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 5, 4 ) ) == expected );

    kernel = KernelFactory( OPERATION_ADDITION, KERNEL_CUDA ).GetKernel();

    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 3 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 3, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 5 ), Matrix_t( 4, 4 ) ) == expected );
    CPPUNIT_ASSERT( *kernel->Operation( Matrix_t( 4, 4 ), Matrix_t( 5, 4 ) ) == expected );
}
