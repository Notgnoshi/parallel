#include "kernels/kernel.h"
#include "kernels/cpu_mult.h"

Kernel::Kernel( ArgumentParser::Operation_e op, ArgumentParser::Kernel_e kernel ) :
    KernelWrapper{GetKernelWrapper( op, kernel )}
{
}

Matrix_t Kernel::Operation( const Matrix_t& lhs, const Matrix_t& rhs )
{
    return KernelWrapper( lhs, rhs );
}

std::function<Matrix_t( const Matrix_t&, const Matrix_t& )> Kernel::GetKernelWrapper( ArgumentParser::Operation_e op, ArgumentParser::Kernel_e kernel )
{
    if( op == ArgumentParser::OPERATION_VECTOR_MULTIPLICATION )
    {
        if( kernel == ArgumentParser::KERNEL_DEFAULT )
        {
        }
        else if( kernel == ArgumentParser::KERNEL_CPU )
        {
        }
    }
    else
    {
        if( kernel == ArgumentParser::KERNEL_DEFAULT )
        {
        }
        else if( kernel == ArgumentParser::KERNEL_CPU )
        {
        }
    }

    //! @todo Implement the rest of the kernels.
    return CpuMultWrapper;
}
