#pragma once

#include "args.h"
#include <memory>

class Kernel;

/**
 * @brief A Kernel factory to create the proper kernel given options.
 */
class KernelFactory
{
public:
    /**
     * @brief Construct a new Kernel object to perform the specified operation
     * with the given kernel.
     *
     * @param op     The operation to perform.
     * @param kernel The kernel to use for the operation.
     */
    KernelFactory( Operation_e op, Kernel_e kernel );

    /**
     * @brief Get the Kernel object from this factory.
     *
     * @returns A shared pointer to the proper kernel.
     */
    std::shared_ptr<Kernel> GetKernel();

private:
    Operation_e operation;
    Kernel_e kernel;
};
