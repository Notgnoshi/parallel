#include "kernels/cpu_mult.h"
#include "matrix.h"

void CpuMultiplicationKernel::Kernel( const Matrix_t& matrix, const Matrix_t& vector, Matrix_t& result )
{
    for( size_t r = 0; r < matrix.rows; ++r )
    {
        // The data field is a contiguous 1D slice.
        result.data[r] = DotProduct( matrix.data + r, vector.data, matrix.cols );
    }
}

std::shared_ptr<Matrix_t> CpuMultiplicationKernel::Operation( const Matrix_t& matrix, const Matrix_t& vector )
{
    auto result = std::make_shared<Matrix_t>( matrix.rows, vector.cols );

    this->Kernel( matrix, vector, *result );

    return result;
}

double CpuMultiplicationKernel::DotProduct( const double* row, const double* col, size_t n )
{
    double sum = 0;
    for( size_t i = 0; i < n; ++i )
    {
        sum += row[i] * col[i];
    }

    return sum;
}
