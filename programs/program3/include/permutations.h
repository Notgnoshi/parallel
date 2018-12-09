#pragma once

#include "common.h"
#include <cstddef>
#include <cstdint>
#include <vector>

/**
 * @brief Get the @f$n@f$th permutation of numbers from 0 to the given length.
 *
 * @details See https://stackoverflow.com/a/7919887/3704977
 *
 * @param length The (exclusive) length.
 * @param n Which permutation to return.
 * @returns The @f$n@f$th permutation.
 */
std::vector<uint8_t> NthPermutation( const size_t length, size_t n );

/**
 * @brief Determines if the given chessboard arrangement is a solution.
 *
 * @details This function assumes that no Queen will be placed in identical rows
 * or columns. That is, each input will be a permutation of the integers 0..n-1.
 *
 * @param arrangement The chessboard arrangement.
 * @returns true if the arrangement is a solution. False otherwise.
 */
bool IsSolution( const std::vector<uint8_t> arrangement );
