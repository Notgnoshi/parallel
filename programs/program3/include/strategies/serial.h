#pragma once
#include "strategies/strategy.h"
#include <vector>

/**
 * @brief Implements a simple serial strategy to solve the @f$n@f$ queens problem.
 */
class SerialStrategy : public Strategy
{
public:
    /**
     * @brief Construct a new Strategy object.
     *
     * @details If the `time` parameter is `true`, the Strategy will time the operation
     * and print the result to stderr in the format `__PRETTY_FUNCTION__: <time> ms`.
     *
     * @param file_output If nonempty, the filename to save the outputs to.
     * @param time Whether or not to time the solution.
     */
    SerialStrategy( std::string file_output = "", bool time = false );

    /**
     * @brief Run the Strategy on a problem of the given size.
     *
     * @param n      The chessboard size.
     * @param screen_output Whether to output solutions to the screen. Defaults to true.
     *
     * @returns The number of solutions found.
     */
    size_t Run( size_t n, bool screen_output = true ) override;

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

private:
    std::string file_output;
    bool time;
};
