#pragma once
#include "strategies/distributed/process.h"

/**
 * @brief A slave Process to do the MasterProcess's bidding.
 */
class SlaveProcess : public Process
{
public:
    /**
     * @brief Construct a new Process object.
     *
     * @param rank This processor's rank. 0 is the master process, while all
     * others are slaves.
     * @param num_procs The total number of processors that are being utilized.
     * @param n The problem size.
     * @param verbose Whether to give diagnostic information. Defaults to false.
     */
    SlaveProcess( size_t rank, size_t num_procs, size_t n, bool verbose = false ) :
        Process( rank, num_procs, n, verbose ),
        uphill( 2 * n - 1, false ),
        downhill( 2 * n - 1, false )
    {}

    /**
     * @brief Perform the work that this slave has been assigned.
     *
     * @returns The number of solutions this slave has found. Must meet the quota
     * or the slave will be sent to a reeducation camp.
     */
    size_t Run( bool screen_output = false, std::string file_output = "" ) override;

    /**
     * @brief Determines if the given arrangement is a solution.
     *
     * @param arrangement A vector of queen placements.
     * @returns True if the given arrangement is a solution. False otherwise.
     */
    bool IsSolution( const std::vector<uint8_t>& arrangement );

    /**
     * @brief Insert a process's rank into a filename.
     *
     * @param rank The rank to inject into the filename.
     * @param filename The filename to inject the rank into.
     * @returns The modified filename.
     */
    static std::string InsertRank( const size_t rank, const std::string& filename );

private:
    std::vector<bool> uphill;
    std::vector<bool> downhill;

    /**
     * @brief Clear the diagonal arrays between checks.
     */
    void ClearDiagonals();
};
