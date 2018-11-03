#include "cpu_mult.h"
#include "matrix.h"

void CpuMultKernelTest::SimpleDotProduct()
{
    const double v1[4] = {1, 1, 1, 1};
    const double v2[4] = {1, 1, 1, 1};
    const double result = 4;

    CPPUNIT_ASSERT( Dot( v1, v2, 4 ) == result );
}

void CpuMultKernelTest::MatrixDotProduct()
{
    Matrix_t v1( 4, 1 );
    Matrix_t v2( 4, 1 );
    for( size_t i = 0; i < 4; ++i )
    {
        v1.data[i] = 1;
        v2.data[i] = 1;
    }
    const double result = 4;

    CPPUNIT_ASSERT( Dot( v1.data, v2.data, v1.rows ) == result );
}

void CpuMultKernelTest::MatVectMult()
{
    const Matrix_t m( "./matrices/4x4_ones.mat" );
    const Matrix_t v( "./matrices/4x1_ones.mat" );
    const Matrix_t r( "./matrices/4x1_fours.mat" );

    Matrix_t result( 4, 1 );

    CpuMultKernel( m, v, result );

    CPPUNIT_ASSERT( result == r );

    const std::shared_ptr<Matrix_t> result2 = CpuMultWrapper( m, v );

    CPPUNIT_ASSERT( *result2 == r );
}
