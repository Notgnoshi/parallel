#pragma once

#include "argument_parser.h"
#include "matrix.h"
#include <functional>

class Kernel
{
public:
    /**
     * @brief Construct a new Kernel object to perform the specified operation
     * with the given kernel.
     *
     * @param op     The operation to perform.
     * @param kernel The kernel to use for the operation.
     */
    Kernel( ArgumentParser::Operation_e op, ArgumentParser::Kernel_e kernel );

    /**
     * @brief Perform the desired operation on the given operands.
     *
     * @param lhs The left operand.
     * @param rhs The right operation.
     * @returns The operation result.
     */
    Matrix_t Operation( const Matrix_t& lhs, const Matrix_t& rhs );

private:
    std::function<Matrix_t( const Matrix_t&, const Matrix_t& )> KernelWrapper;

    /**
     * @brief Get the Kernel Wrapper object to perform the given kernel.
     *
     * @returns The kernel wrapper.
     */
    std::function<Matrix_t( const Matrix_t&, const Matrix_t& )> GetKernelWrapper( ArgumentParser::Operation_e, ArgumentParser::Kernel_e );
};
