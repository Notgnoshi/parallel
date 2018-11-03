#pragma once

struct Matrix_t;

/**
 * @brief Determine if the two matrices can be right-multiplied.
 *
 * @param left  The left multiplication operand
 * @param right The right multiplication operand
 * @returns true if the dimensions are valid, false otherwise.
 */
bool MultiplicationValidator( const Matrix_t& left, const Matrix_t& right );

/**
 * @brief Determine if the two matrices can be added.
 *
 * @param left  The left addition operand
 * @param right The right addition operand
 * @returns true if the dimensions are valid, false otherwise.
 */
bool AdditionValidator( const Matrix_t& left, const Matrix_t& right );
