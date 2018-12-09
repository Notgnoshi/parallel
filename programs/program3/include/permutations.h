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
