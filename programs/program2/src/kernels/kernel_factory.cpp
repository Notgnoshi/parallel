#include "kernels/kernel_factory.h"
#include "kernels/cpu_add.h"
#include "kernels/cpu_mult.h"

KernelFactory::KernelFactory( Operation_e op, Kernel_e kernel ) :
    operation( op ),
    kernel( kernel )
{
}

std::shared_ptr<Kernel> KernelFactory::GetKernel()
{
    switch( this->operation )
    {
    case OPERATION_VECTOR_MULTIPLICATION:
        switch( this->kernel )
        {
        case KERNEL_CPU:
            return std::make_shared<CpuMultiplicationKernel>();

        case KERNEL_DEFAULT:
            return std::make_shared<CpuMultiplicationKernel>();
        }
    case OPERATION_ADDITION:
        switch( this->kernel )
        {
        case KERNEL_CPU:
            return std::make_shared<CpuAdditionKernel>();

        case KERNEL_DEFAULT:
            return std::make_shared<CpuAdditionKernel>();
        }
    }

    // Default kernel to satisfy the compiler.
    return std::make_shared<CpuAdditionKernel>();
}
