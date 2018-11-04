#include "kernel_test.h"
#include "args.h"
#include "kernels/kernel.h"
#include "kernels/kernel_factory.h"
#include "matrix.h"

void KernelTest::DefaultMultOperation()
{
    KernelFactory factory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_DEFAULT );
    auto kernel = factory.GetKernel();

    Matrix_t lhs( "./tests/matrices/4x4_ones.mat" );
    Matrix_t rhs( "./tests/matrices/4x1_ones.mat" );

    const Matrix_t expected( "./tests/matrices/4x1_fours.mat" );

    const std::shared_ptr<Matrix_t> result = kernel->Operation( lhs, rhs );

    CPPUNIT_ASSERT( *result == expected );
}

void KernelTest::DefaultAddOperation()
{
    KernelFactory factory( OPERATION_ADDITION, KERNEL_DEFAULT );
    auto kernel = factory.GetKernel();

    Matrix_t lhs( "./tests/matrices/3x4_ones.mat" );
    Matrix_t rhs( "./tests/matrices/3x4_twos.mat" );

    const Matrix_t expected( "./tests/matrices/3x4_threes.mat" );

    const std::shared_ptr<Matrix_t> result = kernel->Operation( lhs, rhs );

    CPPUNIT_ASSERT( *result == expected );
}
