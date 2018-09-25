#pragma once

#include <stdbool.h>
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
bool circuit_one( uint32_t pid, uint16_t z );

/**
 * @brief
 *
 * @param pid
 * @param z
 */
bool circuit_two( uint32_t pid, uint16_t z );
