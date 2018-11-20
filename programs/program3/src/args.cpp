#include "args.h"
#include <iomanip>
#include <iostream>

ArgumentParser::ArgumentParser( int argc, char** argv ) :
    argc( argc )
{
    for( int i = 0; i < this->argc; ++i )
    {
        this->argv.emplace_back( argv[i] );
    }
}

ArgumentParser::Args_t ArgumentParser::ParseArgs()
{
    Args_t args;
    std::vector<std::string> positionals;

    for( auto arg = this->argv.begin() + 1; arg != this->argv.end(); ++arg )
    {
        if( *arg == "-h" || *arg == "--help" )
        {
            this->Usage();
            exit( 0 );
        }
        else if( *arg == "-o" || *arg == "--output" )
        {
            ++arg;
            args.output = *arg;
        }
        else if( *arg == "-s" || *arg == "--strategy" )
        {
            ++arg;
            try
            {
                args.strategy = static_cast<Strategy_e>( std::stoi( *arg ) );
            } catch( ... )
            {
                std::cerr << "Failed to parse strategy." << std::endl;
                this->Usage();
                exit( 1 );
            }

            if( args.strategy <= STRATEGY_BEGIN || args.strategy >= STRATEGY_END )
            {
                std::cerr << "Invalid strategy given." << std::endl;
                this->Usage();
                exit( 1 );
            }
        }
        else if( *arg == "-t" || *arg == "--time" )
        {
            args.time = true;
        }
        else
        {
            positionals.push_back( *arg );
        }
    }

    if( positionals.size() != 1 )
    {
        std::cerr << "Invalid positional argument(s)." << std::endl;
        this->Usage();
        exit( 1 );
    }

    try
    {
        args.n = std::stoi( positionals[0] );
    } catch( ... )
    {
        std::cerr << "Failed to parse 'n'" << std::endl;
        this->Usage();
        exit( 1 );
    }

    return args;
}

void ArgumentParser::Usage()
{
    std::cout << "Usage: " << this->argv[0] << " ";
    std::cout << "[--help] [--output OUTPUT] [--strategy STRATEGY] [--time]"
              << " ";
    std::cout << "<n>";
    std::cout << std::endl
              << std::endl;
    std::cout << "A distributed memory solution to the n queens problem.";
    std::cout << std::endl
              << std::endl;

    std::cout << "positional arguments:" << std::endl;
    std::cout << std::left << std::setw( 18 ) << "  n"
              << "The length of one side of the chess board." << std::endl;
    std::cout << std::endl
              << std::endl;

    std::cout << "optional arguments:" << std::endl;
    std::cout << std::left << std::setw( 18 ) << " -h, --help"
              << "Show this help message and exit" << std::endl;
    std::cout << std::left << std::setw( 18 ) << " -o, --output"
              << "The filename to print solutions to." << std::endl;
    std::cout << std::left << std::setw( 18 ) << " -s, --strategy"
              << "The strategy to use for the solution. Must be one of" << std::endl;
    std::cout << std::left << std::setw( 18 ) << " "
              << "    1 - Use a slow serial solution strategy." << std::endl;
    std::cout << std::left << std::setw( 18 ) << " "
              << "    2 - Use a slow shared memory parallel strategy." << std::endl;
    std::cout << std::left << std::setw( 18 ) << " "
              << "    3 - Use a distributed memory strategy." << std::endl;
    std::cout << std::left << std::setw( 18 ) << " "
              << "    4 - Use a distributed CUDA strategy." << std::endl;
    std::cout << std::left << std::setw( 18 ) << " -t, --time"
              << "Whether or not to time the solution duration." << std::endl;
}
