#include "strategy_factory.h"
#include "strategies/serial.h"

StrategyFactory::StrategyFactory( ArgumentParser::Args_t args ) :
    args( args )
{
}

std::shared_ptr<Strategy> StrategyFactory::GetStrategy()
{
    switch( this->args.strategy )
    {
    case STRATEGY_SERIAL:
    case STRATEGY_SHARED:
    case STRATEGY_DISTRIBUTED:
    case STRATEGY_DISTRIBUTED_CUDA:
    default:
        return std::make_shared<SerialStrategy>( args.output, args.time );
    }
}
