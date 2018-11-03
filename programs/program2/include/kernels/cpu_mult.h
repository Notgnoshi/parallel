#pragma once

#include "matrix.h"

/**
 * @brief Multiply the given matrix by the given vector on the CPU.
 *
 * @note The dimensions *must* be valid.
 *
 * @param      matrix An r-by-c matrix
 * @param      vector A c-by-1 column vector
 * @param[out] result An r-by-1 column vector
 */
void CpuMultKernel( const Matrix_t& matrix, const Matrix_t& vector, Matrix_t& result );

/**
 * @brief A wrapper for CPU matrix-vector multiplication kernel.
 *
 * @note the dimensions *must* be valid.
 *
 * @param matrix An r-by-c matrix.
 * @param vector A c-by-1 column vector
 * @returns An r-by-1 column vector.
 */
Matrix_t CpuMultWrapper( const Matrix_t& matrix, const Matrix_t& vector );

/**
 * @brief A naive implementation of the vector dot product.
 *
 * @note The dimensions must match, or you'll get some funny results.
 *
 * @param row The left hand vector. Stored contiguously.
 * @param col The right hand vector. Stored contiguously.
 * @param n   The length of the two vector.
 * @returns The vector dot product.
 */
double Dot( const double* row, const double* col, size_t n );
