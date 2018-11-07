#include "cuda_add_test.h"
#include "kernels/kernel.h"
#include "kernels/kernel_factory.h"
#include "matrix.h"

void CudaAdditionKernelTest::SimpleAddition()
{
    Matrix_t lhs( 16, 16 );
    Matrix_t rhs( 16, 16 );
    Matrix_t expected( 16, 16 );

    for( size_t i = 0; i < lhs.elements; ++i )
    {
        lhs.data[i] = 1;
        rhs.data[i] = 2;
        expected.data[i] = lhs.data[i] + rhs.data[i];
    }

    auto kernel = KernelFactory( OPERATION_ADDITION, KERNEL_CUDA ).GetKernel();
    auto result = kernel->Operation( lhs, rhs );

    CPPUNIT_ASSERT( *result == expected );
}

void CudaAdditionKernelTest::LargeAddition()
{
    Matrix_t lhs( 16 * 10, 16 * 10 );
    Matrix_t rhs( 16 * 10, 16 * 10 );
    Matrix_t expected( 16 * 10, 16 * 10 );

    for( size_t i = 0; i < lhs.elements; ++i )
    {
        lhs.data[i] = 1;
        rhs.data[i] = 2;
        expected.data[i] = lhs.data[i] + rhs.data[i];
    }

    auto kernel = KernelFactory( OPERATION_ADDITION, KERNEL_CUDA ).GetKernel();
    auto result = kernel->Operation( lhs, rhs );

    CPPUNIT_ASSERT( *result == expected );
}

void CudaAdditionKernelTest::MismatchedLarger()
{
    Matrix_t lhs( 17, 17 );
    Matrix_t rhs( 17, 17 );
    Matrix_t expected( 17, 17 );

    for( size_t i = 0; i < lhs.elements; ++i )
    {
        lhs.data[i] = 1;
        rhs.data[i] = 2;
        expected.data[i] = lhs.data[i] + rhs.data[i];
    }

    auto kernel = KernelFactory( OPERATION_ADDITION, KERNEL_CUDA ).GetKernel();
    auto result = kernel->Operation( lhs, rhs );

    CPPUNIT_ASSERT( *result == expected );
}

void CudaAdditionKernelTest::MismatchedSmaller()
{
    Matrix_t lhs( 15, 15 );
    Matrix_t rhs( 15, 15 );
    Matrix_t expected( 15, 15 );

    for( size_t i = 0; i < lhs.elements; ++i )
    {
        lhs.data[i] = 1;
        rhs.data[i] = 2;
        expected.data[i] = lhs.data[i] + rhs.data[i];
    }

    auto kernel = KernelFactory( OPERATION_ADDITION, KERNEL_CUDA ).GetKernel();
    auto result = kernel->Operation( lhs, rhs );

    CPPUNIT_ASSERT( *result == expected );
}

void CudaAdditionKernelTest::LargeMismatched()
{
    // Pick matrix dimensions that definitely are not evenly divisible by 16x16.
    static const size_t rows = 873;
    static const size_t cols = 3767;

    Matrix_t lhs( rows, cols );
    Matrix_t rhs( rows, cols );
    Matrix_t expected( rows, cols );
    for( size_t i = 0; i < lhs.elements; ++i )
    {
        lhs.data[i] = static_cast<double>( i ) + 0.5;
        rhs.data[i] = static_cast<double>( i ) + 1.5;
        expected.data[i] = lhs.data[i] + rhs.data[i];
    }

    auto kernel = KernelFactory( OPERATION_ADDITION, KERNEL_CUDA ).GetKernel();
    auto result = kernel->Operation( lhs, rhs );

    CPPUNIT_ASSERT( *result == expected );
}
