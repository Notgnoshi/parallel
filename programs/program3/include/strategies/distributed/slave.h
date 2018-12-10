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
     * @brief Perform the work that this slave has been assigned.
     *
     * @returns The number of solutions this slave has found. Must meet the quota
     * or the slave will be sent to a reeducation camp.
     */
    size_t Run() override;
};
