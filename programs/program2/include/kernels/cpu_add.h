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
     * @details If the `time` parameter is `true`, the Kernel will time the operation
     * and print the result to stderr in the format `__FUNCTION(): <time> ms`.
     *
     * @param lhs The left operand.
     * @param rhs The right operand.
     * @param time Whether or not to time the matrix operation. Defaults to false.
     * @returns A shared pointer to the operation result.
     */
    std::shared_ptr<Matrix_t> Operation( const Matrix_t& lhs, const Matrix_t& rhs, bool time = false ) override;

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
