#pragma once

#include "kernels/kernel.h"
#include <memory>

struct Matrix_t;

/**
 * @brief Implements matrix addition with a CUDA kernel.
 */
class CudaAdditionKernel : public Kernel
{
public:
    /**
     * @brief Calls the CUDA kernel to perform the matrix addition.
     *
     * @note This is a blocking function call that will wait until the CUDA
     * kernel is finished with the operation.
     *
     * @see AdditionKernel for the implementation documentation.
     *
     * @param lhs The left operand.
     * @param rhs The right operand.
     * @returns a shared pointer to the result matrix.
     */
    std::shared_ptr<Matrix_t> Operation( const Matrix_t& lhs, const Matrix_t& rhs ) override;
};
