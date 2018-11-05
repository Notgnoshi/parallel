#include "kernels/kernel_factory.h"
#include "kernels/cpu_add.h"
#include "kernels/cpu_mult.h"
#include "kernels/cuda_add.h"

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
        case KERNEL_DEFAULT:
        case KERNEL_CUDA:
        case KERNEL_CPU:
            return std::make_shared<CpuMultiplicationKernel>();
        }
    case OPERATION_ADDITION:
        switch( this->kernel )
        {
        case KERNEL_DEFAULT:
        case KERNEL_CPU:
            return std::make_shared<CpuAdditionKernel>();

        case KERNEL_CUDA:
            return std::make_shared<CudaAdditionKernel>();
        }
    }

    // Default kernel to satisfy the compiler.
    return std::make_shared<CpuAdditionKernel>();
}
