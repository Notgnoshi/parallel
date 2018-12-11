#include "strategies/distributed/distributed.h"
#include "common.h"
#include "strategies/distributed/process_factory.h"

size_t DistributedStrategy::Run( size_t n )
{
    return GetProcess( this->GetRank(),
                       this->GetProcs(),
                       n,
                       this->GetVerbose() )
        ->MpiRun( this->screen_output, this->file_output );
}
