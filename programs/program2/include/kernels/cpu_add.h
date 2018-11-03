#pragma once

#include "matrix.h"
#include <memory>

/**
 * @brief Add the two matrices on the CPU.
 *
 * @note The dimensions *must* match!
 *
 * @param lhs The left operand.
 * @param rhs The right operand.
 * @param[out] result The addition result.
 */
void CpuAddKernel( const Matrix_t& lhs, const Matrix_t& rhs, Matrix_t& result );

/**
 * @brief A wrapper for CPU matrix addition.
 *
 * @note The dimensions *must* match!
 *
 * @param lhs The left operand.
 * @param rhs The right operand.
 * @returns The addition result.
 */
std::shared_ptr<Matrix_t> CpuAddWrapper( const Matrix_t& lhs, const Matrix_t& rhs );
