#pragma once
#include "permutations.h"
#include "strategies/strategy.h"
#include <omp.h>
#include <vector>

/**
 * @brief Implements a shared memory strategy to solve the @f$n@f$ queens problem.
 */
class SharedStrategy : public Strategy
{
public:
    /**
     * @brief Construct a new Strategy object.
     *
     * @details If the `time` parameter is `true`, the Strategy will time the operation
     * and print the result to stderr in the format `__PRETTY_FUNCTION__: <time> ms`.
     *
     * @param file_output If nonempty, the filename to save the outputs to.
     * @param screen_output Whether to output solutions to the screen. Defaults to false.
     * @param time Whether or not to time the solution. Defaults to false.
     */
    SharedStrategy( std::string file_output = "", bool screen_output = false, bool time = false );

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

private:
    std::string file_output;
    bool screen_output;
    bool time;

    /**
     * @brief Enable per-thread initialization code.
     */
    struct ThreadContext
    {
        /**
         * @brief Construct a new Thread Context object.
         *
         * @details This constructor is called in the main thread and then copied
         * to all of the others.
         *
         * @see https://stackoverflow.com/a/10737658/3704977
         *
         * @param n
         * @param threads
         */
        explicit ThreadContext( size_t n, size_t threads ) :
            n( n ),
            threads( threads ),
            perm( n ),
            uphill( 2 * n - 1, false ),
            downhill( 2 * n - 1, false ) {}

        /**
         * @brief Copy-Construct a new Thread Context object.
         *
         * @details This constructor will be called once at the beginning of each
         * thread. Use this to hold the thread-specific permutation to check.
         *
         * @param ctx The Thread Context object to copy.
         */
        ThreadContext( ThreadContext& ctx ) :
            n( ctx.n ),
            threads( ctx.threads ),
            uphill( ctx.uphill ),
            downhill( ctx.downhill )
        {
            perm = NthPermutation( n, FACTORIALS[n] * omp_get_thread_num() / omp_get_num_procs() );
        }

        size_t n;                   //!> The problem size.
        size_t threads;             //!> The total number of threads being run.
        std::vector<uint8_t> perm;  //!> The thread-specific permutation.
        std::vector<bool> uphill;   //!> Record whether there is a queen in one of the uphill diagonals.
        std::vector<bool> downhill; //!> Record whether there is a queen in one of the downhill diagonals.
    };
};
