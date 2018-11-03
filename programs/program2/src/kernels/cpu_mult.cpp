#include "kernels/cpu_mult.h"

void CpuMultKernel( const Matrix_t& matrix, const Matrix_t& vector, Matrix_t& result )
{
    for( size_t r = 0; r < matrix.rows; ++r )
    {
        // The data field is a contiguous 1D slice.
        result.data[r] = Dot( matrix.data + r, vector.data, matrix.cols );
    }
}

Matrix_t CpuMultWrapper( const Matrix_t& matrix, const Matrix_t& vector )
{
    Matrix_t result( matrix.rows, vector.cols );

    CpuMultKernel( matrix, vector, result );

    return result;
}

double Dot( const double* row, const double* col, size_t n )
{
    double sum = 0;
    for( size_t i = 0; i < n; ++i )
    {
        sum += row[i] * col[i];
    }

    return sum;
}
