#include "strategy_factory.h"
#include "strategies/serial.h"
#include "strategies/shared.h"
#include "strategies/distributed/distributed.h"

StrategyFactory::StrategyFactory( ArgumentParser::Args_t args ) :
    args( args )
{
}

std::shared_ptr<Strategy> StrategyFactory::GetStrategy()
{
    switch( this->args.strategy )
    {
    case STRATEGY_SERIAL:
        return std::make_shared<SerialStrategy>( args.output, args.screen_output, args.time, args.verbose );
    case STRATEGY_SHARED:
        return std::make_shared<SharedStrategy>( args.output, args.screen_output, args.time, args.verbose );
    case STRATEGY_DISTRIBUTED:
    case STRATEGY_DISTRIBUTED_CUDA:
        return std::make_shared<DistributedStrategy>( args.output, args.screen_output, args.time, args.verbose );

    // This is just to satisfy the compiler. The program exits if an invalid strategy is given.
    default:
        return nullptr;
    }
}
