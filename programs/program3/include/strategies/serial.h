#pragma once
#include "permutations.h"
#include "strategies/strategy.h"
#include <omp.h>
#include <vector>

/**
 * @brief Implements a simple serial strategy to solve the @f$n@f$ queens problem.
 */
class SerialStrategy : public Strategy
{
public:
    using Strategy::Strategy;

    /**
     * @brief Run the Strategy on a problem of the given size.
     *
     * @param n      The chessboard size.
     *
     * @returns The number of solutions found.
     */
    size_t Run( size_t n ) override;

    /**
     * @brief Is the given arrangement a solution?
     *
     * @details I pass in the downhill and uphill vectors to make this function
     * avoid reallocating them over and over, yet still allow thread-safety.
     *
     * @param arrangement
     * @param downhill Vector of downhill diagonal placements.
     * @param uphill Vector of uphill diagonal placements.
     * @returns True if the given arrangement is a solution. False otherwise.
     */
    static bool IsSolution( const std::vector<uint8_t>& arrangement, std::vector<bool>& downhill, std::vector<bool>& uphill );

    /**
     * @brief Clears and zeros the given vector.
     *
     * @param v The vector to clear.
     */
    static void ClearVector( std::vector<bool>& v );
};
