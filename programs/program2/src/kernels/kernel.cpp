#include "kernels/kernel.h"
#include "kernels/cpu_add.h"
#include "kernels/cpu_mult.h"
#include "validator.h"
#include <iostream>

Kernel::Kernel( ArgumentParser::Operation_e op, ArgumentParser::Kernel_e kernel ) :
    operation( op ),
    kernel( kernel )
{
}

std::shared_ptr<Matrix_t> Kernel::Operation( const Matrix_t& lhs, const Matrix_t& rhs )
{
    if( this->operation == ArgumentParser::OPERATION_VECTOR_MULTIPLICATION )
    {
        if( MultiplicationValidator( lhs, rhs ) )
        {
            return GetKernelWrapper()( lhs, rhs );
        }
    }
    else if( this->operation == ArgumentParser::OPERATION_ADDITION )
    {
        if( AdditionValidator( lhs, rhs ) )
        {
            return GetKernelWrapper()( lhs, rhs );
        }
    }

    std::cerr << "Invalid matrix dimensions" << std::endl;
    return std::make_shared<Matrix_t>( 0, 0 );
}

std::function<std::shared_ptr<Matrix_t>( const Matrix_t&, const Matrix_t& )> Kernel::GetKernelWrapper()
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
