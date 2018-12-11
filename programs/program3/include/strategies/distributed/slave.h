#pragma once
#include "strategies/distributed/process.h"

/**
 * @brief A slave Process to do the MasterProcess's bidding.
 */
class SlaveProcess : public Process
{
public:
    using Process::Process;

    /**
     * @brief Run the slave process.
     *
     * @details The slave processes receive a work assignment and diligently,
     * well.. slave over their work until done. It was their own fault for not
     * unionizing.
     *
     * @param screen_output Whether to print solutions to the console.
     * @param file_output An optional filename to print solutions to. The process
     * rank will be appended to the filename.
     *
     * @returns All of the slave process return the number of solutions that
     * they find.
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
};
