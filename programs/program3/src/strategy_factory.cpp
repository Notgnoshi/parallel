#include "strategy_factory.h"
#include "strategies/serial.h"
#include "strategies/shared.h"

StrategyFactory::StrategyFactory( ArgumentParser::Args_t args ) :
    args( args )
{
}

std::shared_ptr<Strategy> StrategyFactory::GetStrategy()
{
    switch( this->args.strategy )
    {
    case STRATEGY_SERIAL:
        return std::make_shared<SerialStrategy>( args.output, args.screen_output, args.time );
    case STRATEGY_SHARED:
        return std::make_shared<SharedStrategy>( args.output, args.screen_output, args.time );
    case STRATEGY_DISTRIBUTED:
    case STRATEGY_DISTRIBUTED_CUDA:
        return std::make_shared<SharedStrategy>( args.output, args.screen_output, args.time );

    // This is just to satisfy the compiler. The program exits if an invalid strategy is given.
    default:
        return nullptr;
    }
}
