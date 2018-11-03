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
     * @brief Perform the Kernel's operation on the given operands.
     *
     * @note If the given dimensions are invalid, the result will not be computed,
     * and an empty Matrix_t of size (0, 0) will be returned.
     *
     * @todo Validate the operand sizes in the implementations.
     *
     * @param lhs The left Matrix_t operand.
     * @param rhs The right Matrix_t operand.
     * @returns A shared pointer to the Matrix_t operation result.
     */
    virtual std::shared_ptr<Matrix_t> Operation( const Matrix_t& lhs, const Matrix_t& rhs ) = 0;
};
