#pragma once

/**
 * @brief Adds two arrays of the given size.
 *
 * @param[in, out] a  The array to increment and update.
 * @param[in] b       The array to add to a
 * @param n           The size of the two arrays.
 */
__global__ void incr_kernel( double* a, const double* b, size_t n );
