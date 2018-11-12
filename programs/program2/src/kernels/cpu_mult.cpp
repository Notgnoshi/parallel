#include "kernels/cpu_mult.h"
#include "matrix.h"
#include "validator.h"
#include <iostream>
#include <sys/time.h>

void CpuMultiplicationKernel::Kernel( const Matrix_t& matrix, const Matrix_t& vector, Matrix_t& result )
{
    for( size_t r = 0; r < matrix.rows; ++r )
    {
        // The data field is a contiguous 1D slice.
        result.data[r] = DotProduct( matrix.data + r * matrix.cols, vector.data, matrix.cols );
    }
}

std::shared_ptr<Matrix_t> CpuMultiplicationKernel::Operation( const Matrix_t& matrix, const Matrix_t& vector, bool time )
{
    struct timeval begin, end;
    double duration;

    if( !MultiplicationValidator( matrix, vector ) )
    {
        // std::cerr << "Dimensions (" << matrix.rows << ", " << matrix.cols << ")"
        //           << " * (" << vector.rows << ", " << vector.cols << ")"
        //           << " incompatible for multiplication." << std::endl;
        return std::make_shared<Matrix_t>( 0, 0 );
    }

    auto result = std::make_shared<Matrix_t>( matrix.rows, vector.cols );

    gettimeofday( &begin, nullptr );

    this->Kernel( matrix, vector, *result );

    gettimeofday( &end, nullptr );
    if( end.tv_usec < begin.tv_usec )
    {
        end.tv_usec += 1000000;
        begin.tv_sec += 1;
    }

    duration = static_cast<double>( end.tv_sec - begin.tv_sec ) + static_cast<double>( end.tv_usec - begin.tv_usec ) / 1000.0;

    if( time )
    {
        std::cerr << __PRETTY_FUNCTION__ << ": " << duration << " ms" << std::endl;
    }

    return result;
}

float CpuMultiplicationKernel::DotProduct( const float* row, const float* col, size_t n )
{
    float sum = 0;
    for( size_t i = 0; i < n; ++i )
    {
        sum += row[i] * col[i];
    }

    return sum;
}
