#pragma once
#include "strategies/distributed/process.h"

/**
 * @brief A master Process for giving the SlaveProcesses orders.
 */
class MasterProcess : public Process
{
public:
    using Process::Process;

    /**
     * @brief Assign work and receive reports from the SlaveProcesses.
     *
     * @returns The total number of solutions found.
     */
    size_t Run( bool screen_output = false, std::string file_output = "" ) override;
};
