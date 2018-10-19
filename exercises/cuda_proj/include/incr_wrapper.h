#pragma once
#include <cstddef>
#include <cstdint>

/**
 * @brief GPU accelerated vector increment.
 *
 * @param[in, out] a The vector to increment.
 * @param[in]      b The vector to add to a.
 * @param          n The size of both vectors.
 */
void incr( double* a, const double* b, size_t n );
