#include "cuda_mult_test.h"
#include "kernels/kernel.h"
#include "kernels/kernel_factory.h"
#include "matrix.h"

void CudaMultiplicationKernelTest::SimpleSmall()
{
    // Matrix_t lhs( 16, 16 );
    // Matrix_t rhs( 16, 1 );

    // for( size_t i = 0; i < lhs.elements; ++i )
    // {
    //     lhs.data[i] = static_cast<double>( i );
    // }

    // for( size_t i = 0; i < rhs.elements; ++i )
    // {
    //     rhs.data[i] = static_cast<double>( i ) + 1.5;
    // }

    // auto cpu_kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CPU ).GetKernel();
    // auto expected = cpu_kernel->Operation( lhs, rhs );

    // auto kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CUDA ).GetKernel();
    // auto result = kernel->Operation( lhs, rhs );

    // CPPUNIT_ASSERT( *result == *expected );
}

void CudaMultiplicationKernelTest::SimpleSquare()
{
    Matrix_t lhs( 16, 16 );
    Matrix_t rhs( 16, 16 );
    Matrix_t expected( 16, 16 );

    for( size_t i = 0; i < lhs.elements; ++i )
    {
        lhs.data[i] = 1;
        rhs.data[i] = 1;
        expected.data[i] = 16;
    }

    auto kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CUDA ).GetKernel();
    auto result = kernel->Operation( lhs, rhs );

    result->Print();

    CPPUNIT_ASSERT( *result == expected );
}

void CudaMultiplicationKernelTest::MismatchedSmaller()
{
    // Matrix_t lhs( 15, 15 );
    // Matrix_t rhs( 15, 1 );

    // for( size_t i = 0; i < lhs.elements; ++i )
    // {
    //     lhs.data[i] = static_cast<double>( i );
    // }

    // for( size_t i = 0; i < rhs.elements; ++i )
    // {
    //     rhs.data[i] = static_cast<double>( i ) + 1.5;
    // }

    // auto cpu_kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CPU ).GetKernel();
    // auto expected = cpu_kernel->Operation( lhs, rhs );

    // auto kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CUDA ).GetKernel();
    // auto result = kernel->Operation( lhs, rhs );

    // CPPUNIT_ASSERT( *result == *expected );
}

void CudaMultiplicationKernelTest::MismatchedBigger()
{
    // Matrix_t lhs( 17, 17 );
    // Matrix_t rhs( 17, 1 );

    // for( size_t i = 0; i < lhs.elements; ++i )
    // {
    //     lhs.data[i] = static_cast<double>( i );
    // }

    // for( size_t i = 0; i < rhs.elements; ++i )
    // {
    //     rhs.data[i] = static_cast<double>( i ) + 1.5;
    // }

    // auto cpu_kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CPU ).GetKernel();
    // auto expected = cpu_kernel->Operation( lhs, rhs );

    // auto kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CUDA ).GetKernel();
    // auto result = kernel->Operation( lhs, rhs );

    // CPPUNIT_ASSERT( *result == *expected );
}

void CudaMultiplicationKernelTest::SimpleLarge()
{
    // Matrix_t lhs( 16 * 10, 16 * 10 );
    // Matrix_t rhs( 16 * 10, 1 );

    // for( size_t i = 0; i < lhs.elements; ++i )
    // {
    //     lhs.data[i] = static_cast<double>( i );
    // }

    // for( size_t i = 0; i < rhs.elements; ++i )
    // {
    //     rhs.data[i] = static_cast<double>( i ) + 1.5;
    // }

    // auto cpu_kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CPU ).GetKernel();
    // auto expected = cpu_kernel->Operation( lhs, rhs );

    // auto kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CUDA ).GetKernel();
    // auto result = kernel->Operation( lhs, rhs );

    // CPPUNIT_ASSERT( *result == *expected );
}

void CudaMultiplicationKernelTest::LargeMismatched()
{
    // Matrix_t lhs( 89, 1237 );
    // Matrix_t rhs( 1237, 1 );

    // for( size_t i = 0; i < lhs.elements; ++i )
    // {
    //     lhs.data[i] = static_cast<double>( i );
    // }

    // for( size_t i = 0; i < rhs.elements; ++i )
    // {
    //     rhs.data[i] = static_cast<double>( i ) + 1.5;
    // }

    // auto cpu_kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CPU ).GetKernel();
    // auto expected = cpu_kernel->Operation( lhs, rhs );

    // auto kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CUDA ).GetKernel();
    // auto result = kernel->Operation( lhs, rhs );

    // CPPUNIT_ASSERT( *result == *expected );
}
