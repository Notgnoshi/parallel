#pragma once

#include <memory>

struct Matrix_t;

/**
 * @brief An abstract base kernel to define the Kernel interface.
 */
class Kernel
{
public:
    /**
     * @brief Default-construct a new Kernel object.
     */
    Kernel() = default;

    /**
     * @brief Default-destroy the Kernel object.
     */
    virtual ~Kernel() = default;

    /**
     * @brief Perform the Kernel's operation on the given operands.
     *
     * @note If the given dimensions are invalid, the result will not be computed,
     * and an empty Matrix_t of size (0, 0) will be returned.
     *
     * @details If the `time` parameter is `true`, the Kernel will time the operation
     * and print the result to stderr in the format `__PRETTY_FUNCTION__: <time> ms`.
     *
     * @param lhs The left Matrix_t operand.
     * @param rhs The right Matrix_t operand.
     * @param time Whether or not to time the matrix operation. Defaults to false.
     * @returns A shared pointer to the Matrix_t operation result.
     */
    virtual std::shared_ptr<Matrix_t> Operation( const Matrix_t& lhs, const Matrix_t& rhs, bool time = false ) = 0;
};
