#pragma once

#include <chrono>
#include <cstddef>
#include <iostream>
#include <string>
#include <vector>

const std::vector<size_t> SOLUTIONS = {
    0,
    1,
    0,
    0,
    2,
    10,
    4,
    40,
    92,
    352,
    724,
    2680,
    14200,
    73712,
    365596,
    2279184,
    14772512,
    95815104,
    666090624,
    4968057848,
    39029188884,
    314666222712,
    2691008701644,
    24233937684440,
    227514171973736,
    2207893435808350,
    22317699619364000,
};

const std::vector<size_t> FACTORIALS = {
    1,
    1,
    2,
    6,
    24,
    120,
    720,
    5040,
    40320,
    362880,
    3628800,
    39916800,
    479001600,
    6227020800,
    87178291200,
    1307674368000,
    20922789888000,
    355687428096000,
    6402373705728000,
    121645100408832000,
    2432902008176640000,
};

/**
 * @brief Append the given block of solutions to the end of the given filename.
 *
 * @details Avoid the need to output individual solutions to save I/O.
 *
 * @param block An arbitrary-length block of solutions to output.
 * @param filename The filename to write the solutions to.
 */
void AppendBlock( std::vector<std::vector<uint8_t>> block, std::string filename );

/**
 * @brief Print the given solution to the screen.
 *
 * @param solution The solution to print.
 */
void PrintSolution( std::vector<uint8_t> solution );

/**
 * @brief A handy-dandy timer class to eliminate soul-crushing copy-pasting.
 *
 * @details Taken from https://stackoverflow.com/a/19471595/3704977
 */
class Timer
{
public:
    /**
     * @brief Start a timer.
     */
    Timer() :
        beg_( clock_::now() ) {}

    /**
     * @brief Reset the timer.
     */
    void reset()
    {
        beg_ = clock_::now();
    }

    /**
     * @brief Get the elapsed time since the timer creation or reset.
     *
     * @returns The number of elapsed seconds.
     */
    double elapsed() const
    {
        return std::chrono::duration_cast<second_>( clock_::now() - beg_ ).count();
    }

private:
    typedef std::chrono::high_resolution_clock clock_;
    typedef std::chrono::duration<double, std::ratio<1>> second_;
    std::chrono::time_point<clock_> beg_;
};
