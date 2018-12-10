#include "strategies/distributed/process_factory.h"
#include "strategies/distributed/master.h"
#include "strategies/distributed/slave.h"

//! @todo Is this function really that necessary?
std::unique_ptr<Process> GetProcess( size_t rank, size_t num_procs, size_t n, bool verbose )
{
    if( rank == 0 )
    {
        return std::make_unique<MasterProcess>( rank, num_procs, n, verbose );
    }

    return std::make_unique<SlaveProcess>( rank, num_procs, n, verbose );
}
