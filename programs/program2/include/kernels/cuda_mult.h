#pragma once

#include "kernels/kernel.h"
#include <memory>

struct Matrix_t;

/**
 * @brief Implements matrix vector multiplication with a CUDA kernel.
 */
class CudaMultiplicationKernel : public Kernel
{
public:
    /**
     * @brief Calls the CUDA kernel to perform the matrix vector multiplication.
     *
     * @note This is a blocking function call that will wait until the CUDA
     * kernel is finished with the operation.
     *
     * @see MultiplicationKernel for the implementation documentation.
     *
     * @param lhs The left operand.
     * @param rhs The right operand.
     * @returns a shared pointer to the result matrix.
     */
    std::shared_ptr<Matrix_t> Operation( const Matrix_t& lhs, const Matrix_t& rhs ) override;
};
