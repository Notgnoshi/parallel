#include "cpu_mult_test.h"
#include "kernels/kernel.h"
#include "kernels/kernel_factory.h"
#include "matrix.h"

void CpuMultKernelTest::MatVectMult()
{
    KernelFactory factory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CPU );
    auto kernel = factory.GetKernel();

    const Matrix_t m( "./tests/matrices/4x4_ones.mat" );
    const Matrix_t v( "./tests/matrices/4x1_ones.mat" );
    const Matrix_t r( "./tests/matrices/4x1_fours.mat" );

    auto result = kernel->Operation( m, v );

    CPPUNIT_ASSERT( *result == r );
}

void CpuMultKernelTest::LessSimple()
{
    auto kernel = KernelFactory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_CPU ).GetKernel();

    const Matrix_t m( "./tests/matrices/4x4_seq.mat" );
    const Matrix_t v( "./tests/matrices/4x1_ones.mat" );
    const Matrix_t r( "./tests/matrices/4x1_seq.ones.mat" );

    auto result = kernel->Operation( m, v );

    // result->Print();

    CPPUNIT_ASSERT( *result == r );
}
