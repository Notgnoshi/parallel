#include "args.h"
#include "strategy_factory.h"
#include <iostream>

/**
 * @brief Program main entry point.
 *
 * Parses commandline arguments, initializes communications, etc.
 *
 * @param argc The usual number of commandline arguments.
 * @param argv The usual array of commandline arguments.
 *
 * @returns 0 if program exited successfully, 1 otherwise.
 */
int main( int argc, const char** argv )
{
    ArgumentParser::Args_t args = ArgumentParser( argc, argv ).ParseArgs();

    // Get the right solution strategy based on the commandline arguments.
    auto strategy = StrategyFactory( args ).GetStrategy();

    // Initialize MPI communications, etc.
    strategy->Initialize( &argc, &argv );

    if( strategy->GetRank() == 0 && args.verbose )
    {
        ArgumentParser::Summarize( args );
    }

    // Get the number of solutions and print them out.
    size_t solutions = strategy->Run( args.n );
    if( strategy->GetRank() == 0 )
    {
        std::cout << "Found " << solutions << " solutions." << std::endl;
    }
    else if( args.verbose )
    {
        std::cout << "Slave " << strategy->GetRank() << " found " << solutions
                  << " solutions." << std::endl;
    }

    // Clean up after MPI communications, etc.
    strategy->Finalize();

    return 0;
}
