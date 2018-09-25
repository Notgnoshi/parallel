#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/**
 * @brief Get individual bits from a number
 *
 * @param n The number to extract bits from
 * @param i The index of the bit to extract, taken from the LSB (index 0)
 */
#define EXTRACT_BIT( n, i ) ( ( n & ( 0x1 << i ) ) ? 0x1 : 0x0 )

/**
 * @brief
 *
 * @param pid
 * @param z
 */
bool circuit_one( const uint32_t pid, const uint16_t z );

/**
 * @brief
 *
 * @param pid
 * @param z
 */
bool circuit_two( const uint32_t pid, const uint16_t z );

/**
 * @brief Extract the bits of a uint16_t into the given boolean array.
 *
 * @param bits An (at least) 16 element boolean array, previously allocated.
 * @param z The number to extract.
 */
static inline void extract_bits( bool* bits, const uint16_t z );
