#include "strategies/distributed/distributed.h"
#include "common.h"
#include "strategies/distributed/process_factory.h"

size_t DistributedStrategy::Run( size_t n )
{
    //! @todo Hack verbosity into the works.
    return GetProcess( this->GetRank(), this->GetProcs(), n, this->GetVerbose() )->Run();
}
