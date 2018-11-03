#include "kernels/cpu_add.h"
#include "matrix.h"

void CpuAdditionKernel::Kernel( const Matrix_t& lhs, const Matrix_t& rhs, Matrix_t& result )
{
    for( size_t r = 0; r < lhs.rows; ++r )
    {
        for( size_t c = 0; c < lhs.cols; ++c )
        {
            // We're going to pretend like I never tried to cast away the=
            // const-ness of the Matrix_t's before I implemented a const-version
            // of operator().
            result( r, c ) = lhs( r, c ) + rhs( r, c );
        }
    }
}

std::shared_ptr<Matrix_t> CpuAdditionKernel::Operation( const Matrix_t& lhs, const Matrix_t& rhs )
{
    auto result = std::make_shared<Matrix_t>( lhs.rows, lhs.cols );

    this->Kernel( lhs, rhs, *result );

    return result;
}
