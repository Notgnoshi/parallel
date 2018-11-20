#pragma once

#include <cstddef>
#include <string>

/**
 * @brief An abstract base class to represent a solution strategy to the @f$n@f$ queens problem.
 */
class Strategy
{
public:
    /**
     * @brief Default-destroy the Strategy object.
     */
    virtual ~Strategy() = default;

    /**
     * @brief Run the Strategy on a problem of the given size.
     *
     * @param n      The chessboard size.
     * @param output Whether to output solutions to the screen. Defaults to true.
     *
     * @returns The number of solutions found.
     */
    virtual size_t Run( size_t n, bool output = true ) = 0;
};
