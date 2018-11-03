#include "kernels/kernel.h"
#include "kernels/cpu_add.h"
#include "kernels/cpu_mult.h"

Kernel::Kernel( ArgumentParser::Operation_e op, ArgumentParser::Kernel_e kernel ) :
    operation( op ),
    kernel( kernel )
{
}

Matrix_t Kernel::Operation( const Matrix_t& lhs, const Matrix_t& rhs )
{
    //! @todo Validate the matrices!!
    return GetKernelWrapper()( lhs, rhs );
}

std::function<Matrix_t( const Matrix_t&, const Matrix_t& )> Kernel::GetKernelWrapper()
{
    //! @todo Implement the rest of the kernels.
    if( this->operation == ArgumentParser::OPERATION_VECTOR_MULTIPLICATION )
    {
        if( this->kernel == ArgumentParser::KERNEL_DEFAULT )
        {
        }
        else if( this->kernel == ArgumentParser::KERNEL_CPU )
        {
        }
        return CpuMultWrapper;
    }
    else
    {
        if( this->kernel == ArgumentParser::KERNEL_DEFAULT )
        {
        }
        else if( this->kernel == ArgumentParser::KERNEL_CPU )
        {
        }

        return CpuAddWrapper;
    }
}
