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

    // result->Print();

    CPPUNIT_ASSERT( *result == expected );
}
