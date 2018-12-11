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
     * @brief Run the Master process.
     *
     * @details The master process does not work, so this is function is only
     * included because it inherits from Process.
     *
     * @param screen_output Whether to print solutions to the console.
     * @param file_output An optional filename to print solutions to. The process
     * rank will be appended to the filename.
     *
     * @returns 0
     */
    size_t Run( bool screen_output = false, std::string file_output = "" ) override;
};
