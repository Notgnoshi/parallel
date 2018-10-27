#include "kernels/cpu_mult.h"

void CpuMultKernel( const Matrix_t& matrix, const Matrix_t& vector, Matrix_t& result )
{
    //! @todo Implement, possibly naively.
    //! @todo Should I implement full matrix-matrix multiplication?
    //! @todo Unit test!!
    (void)matrix;
    (void)vector;
    (void)result;
}

Matrix_t CpuMultWrapper( const Matrix_t& matrix, const Matrix_t& vector )
{
    Matrix_t result( matrix.rows, vector.cols );

    CpuMultKernel( matrix, vector, result );

    return result;
}
