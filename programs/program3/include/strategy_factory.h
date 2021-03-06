#include "args.h"
#include "strategies/strategy.h"
#include <memory>

/**
 * @brief A Strategy Factory to produce a Strategy given commandline options.
 */
class StrategyFactory
{
public:
    /**
     * @brief Construct a new Strategy Factory object from commandline arguments.
     *
     * @param args The commandline arguments to use.
     */
    StrategyFactory( ArgumentParser::Args_t args );

    /**
     * @brief Construct the proper Strategy for the given commandline arguments.
     *
     * @returns a shared pointer to the constructed Strategy.
     */
    std::shared_ptr<Strategy> GetStrategy();

private:
    ArgumentParser::Args_t args;
};
