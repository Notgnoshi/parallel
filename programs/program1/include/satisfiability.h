/**
 * @brief Usual header file for satisfiability functions.
 *
 * @file satisfiability.h
 * @author Austin Gill
 * @date 2018-09-24
 */
#pragma once

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

/**
 * @brief Get individual bits from a number
 *
 * @param n The number to extract bits from
 * @param i The index of the bit to extract, taken from the LSB (index 0)
 */
#define EXTRACT_BIT( n, i ) ( ( n & ( 0x1 << i ) ) ? 0x1 : 0x0 )

/**
 * @brief Checks the given input against the first circuit for satisfiability.
 *
 * @param pid The process ID of the calling thread.
 * @param z The input to check.
 *
 * @returns true if the circuit is satisfied. False otherwise.
 */
bool circuit_one( const uint32_t pid, const uint16_t z );

/**
 * @brief Checks the given input against the second circuit for satisfiability.
 *
 * @param pid The process ID of the calling thread.
 * @param z The input to check.
 *
 * @returns true if the circuit is satisfied. False otherwise.
 */
bool circuit_two( const uint32_t pid, const uint16_t z );

/**
 * @brief Extract the bits of a uint16_t into the given boolean array.
 *
 * @param bits An (at least) 16 element boolean array, previously allocated.
 * @param z The number to extract.
 */
static inline void extract_bits( bool* bits, const uint16_t z )
{
    for( size_t i = 0; i < 16; ++i )
    {
        bits[i] = EXTRACT_BIT( z, i );
    }
}

/**
 * @brief Brute force check the satisfiability of the given circuit.
 *
 * @todo Need lots of documentation, because this is the heavy lifter.
 *
 * @param circuit_fp A function pointer to the circuit to check.
 */
void check_circuit( bool ( *circuit_fp )( const uint32_t, const uint16_t ) );
