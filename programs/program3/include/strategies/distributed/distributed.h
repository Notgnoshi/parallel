#pragma once
#include "strategies/strategy.h"

/**
 * @brief Implements a distributed memory strategy to solve the @f$n@f$ queens problem.
 */
class DistributedStrategy : public Strategy
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

private:
    std::string file_output;
    bool screen_output;
    bool time;
};
