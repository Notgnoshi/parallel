#pragma once
#include "strategies/strategy.h"

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
     * @param output If nonempty, the filename to save the outputs to.
     * @param time Whether or not to time the solution.
     */
    SerialStrategy( std::string output = "", bool time = false );

    /**
     * @brief Run the Strategy on a problem of the given size.
     *
     * @param n      The chessboard size.
     * @param output Whether to output solutions to the screen. Defaults to true.
     *
     * @returns The number of solutions found.
     */
    size_t Run( size_t n, bool output = true ) override;

private:
    std::string output;
    bool time;
};
