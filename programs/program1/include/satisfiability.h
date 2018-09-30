/**
 * @file satisfiability.h
 * @author Austin Gill (atgill@protonmail.com)
 * @brief Usual header file for satisfiability functions.
 */
#pragma once

#include <limits.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

#include <omp.h>

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
 * @param tid The thread ID of the calling thread.
 * @param z The input to check.
 *
 * @returns true if the circuit is satisfied. False otherwise.
 */
bool circuit_one( const int32_t tid, const uint16_t z );

/**
 * @brief Checks the given input against the second circuit for satisfiability.
 *
 * @param tid The thread ID of the calling thread.
 * @param z The input to check.
 *
 * @returns true if the circuit is satisfied. False otherwise.
 */
bool circuit_two( const int32_t tid, const uint16_t z );

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
 * Because there is no dependency from one input to the circuit and another,
 * this function is an excellent candidate for parallelization.
 *
 * This function will split the inputs over @f$p@f$ threads checking the satisfiability
 * of the given circuit in parallel, where @f$p@f$ is the number of processors
 * on the system.
 *
 * @note If compiled with `-DSCHEDULE_COMPARISON`, this function will time and
 * compare the default schedule with static and dynamic schedules with a chunk
 * size of 1. \n
 * The provided Makefile uses `-DSCHEDULE_COMPARISON` by default.
 *
 * @bug When using a static schedule, with a chunk size of 1, and an `uint16_t`
 * index variable, my parallel for loop does not terminate. It works as expected
 * with the default chunk size, or when run sequentially.
 * \n
 * After much DuckDuckGo-ing, I found that earlier versions of OpenMP (specifically
 * version 2.0) required the index variable to be a signed integral type. This
 * requirement was dropped in version 3.0. After checking my OpenMP version with
 * `echo |cpp -fopenmp -dM |grep -i open` I verified I'm using version 4.5.
 * \n
 * This led me to attempt using a `int32_t` index variable, which worked as expected.
 * Interestingly, using `uint32_t` also works.
 * \n
 * See https://stackoverflow.com/a/2822064
 *
 * @param circuit_fp A function pointer to the circuit to check.
 *        The function prototype should be
 *        @code
 *             bool circuit( const int32_t tid, const uint16_t z )
 *        @endcode
 *        where `tid` is the thread ID of the calling thread and `z` is the input
 *        to test for satisfaction.
 */
void check_circuit( bool ( *circuit_fp )( const int32_t, const uint16_t ) );
