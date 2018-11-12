#pragma once

#include "kernels/kernel.h"
#include <memory>

struct Matrix_t;

/**
 * @brief A Kernel to perform matrix-vector multiplication on the CPU.
 */
class CpuMultiplicationKernel : public Kernel
{
public:
    /**
     * @brief Perform the matrix-vector multiplication.
     *
     * @note If the operand dimensions are mismatched, and empty Matrix_t will
     * be returned.
     *
     * @param matrix An r-by-c matrix.
     * @param vector A c-by-1 column vector
     * @returns An r-by-1 column vector.
     */
    std::shared_ptr<Matrix_t> Operation( const Matrix_t& matrix, const Matrix_t& vector ) override;

private:
    /**
     * @brief Multiply the given matrix by the given vector on the CPU.
     *
     * @note The dimensions *must* be valid.
     *
     * @param      matrix An r-by-c matrix
     * @param      vector A c-by-1 column vector
     * @param[out] result An r-by-1 column vector
     */
    void Kernel( const Matrix_t& matrix, const Matrix_t& vector, Matrix_t& result );

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
    static float DotProduct( const float* row, const float* col, size_t n );
};
