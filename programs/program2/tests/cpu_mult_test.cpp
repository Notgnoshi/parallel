#include "cpu_mult_test.h"
#include "kernels/kernel.h"
#include "kernels/kernel_factory.h"
#include "matrix.h"

void CpuMultKernelTest::MatVectMult()
{
    KernelFactory factory( OPERATION_VECTOR_MULTIPLICATION, KERNEL_DEFAULT );
    auto kernel = factory.GetKernel();

    const Matrix_t m( "./tests/matrices/4x4_ones.mat" );
    const Matrix_t v( "./tests/matrices/4x1_ones.mat" );
    const Matrix_t r( "./tests/matrices/4x1_fours.mat" );

    auto result = kernel->Operation( m, v );

    CPPUNIT_ASSERT( *result == r );
}
