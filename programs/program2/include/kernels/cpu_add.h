#pragma once

#include "kernels/kernel.h"
#include <memory>

struct Matrix_t;

/**
 * @brief A Kernel to implement matrix addition on the CPU.
 */
class CpuAdditionKernel : public Kernel
{
public:
    /**
     * @brief Perform the matrix addition.
     *
     * @note If the operand dimensions are mismatched, and empty Matrix_t will
     * be returned.
     *
     * @param lhs The left operand.
     * @param rhs The right operand.
     * @returns A shared pointer to the operation result.
     */
    std::shared_ptr<Matrix_t> Operation( const Matrix_t& lhs, const Matrix_t& rhs ) override;

private:
    /**
     * @brief Add the two matrices on the CPU.
     *
     * @note The dimensions *must* match!
     *
     * @param lhs The left operand.
     * @param rhs The right operand.
     * @param[out] result The addition result.
     */
    void Kernel( const Matrix_t& lhs, const Matrix_t& rhs, Matrix_t& result );
};
