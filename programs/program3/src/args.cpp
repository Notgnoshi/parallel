#include "args.h"
#include <iomanip>
#include <iostream>

using namespace std;

ArgumentParser::ArgumentParser( int argc, const char** argv ) :
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
    vector<string> positionals;

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
                args.strategy = static_cast<Strategy_e>( stoi( *arg ) );
            } catch( ... )
            {
                cerr << "Failed to parse strategy." << endl;
                this->Usage();
                exit( 1 );
            }

            if( args.strategy <= STRATEGY_BEGIN || args.strategy >= STRATEGY_END )
            {
                cerr << "Invalid strategy given." << endl;
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
        cerr << "Invalid positional argument(s)." << endl;
        this->Usage();
        exit( 1 );
    }

    try
    {
        args.n = stoi( positionals[0] );
    } catch( ... )
    {
        cerr << "Failed to parse 'n'" << endl;
        this->Usage();
        exit( 1 );
    }

    return args;
}

void ArgumentParser::Usage()
{
    cout << "Usage: " << this->argv[0] << " ";
    cout << "[--help] [--output OUTPUT] [--strategy STRATEGY] [--time]"
         << " ";
    cout << "<n>";
    cout << endl
         << endl;
    cout << "A distributed memory solution to the n queens problem.";
    cout << endl
         << endl;

    cout << "positional arguments:" << endl;
    cout << left << setw( 18 ) << "  n"
         << "The length of one side of the chess board." << endl;
    cout << endl
         << endl;

    cout << "optional arguments:" << endl;
    cout << left << setw( 18 ) << " -h, --help"
         << "Show this help message and exit" << endl;
    cout << left << setw( 18 ) << " -o, --output"
         << "The filename to print solutions to." << endl;
    cout << left << setw( 18 ) << " -s, --strategy"
         << "The strategy to use for the solution. Must be one of" << endl;
    cout << left << setw( 18 ) << " "
         << "    1 - Use a slow serial solution strategy." << endl;
    cout << left << setw( 18 ) << " "
         << "    2 - Use a slow shared memory parallel strategy." << endl;
    cout << left << setw( 18 ) << " "
         << "    3 - Use a distributed memory strategy." << endl;
    cout << left << setw( 18 ) << " "
         << "    4 - Use a distributed CUDA strategy." << endl;
    cout << left << setw( 18 ) << " -t, --time"
         << "Whether or not to time the solution duration." << endl;
}

void ArgumentParser::Summarize( Args_t args )
{
    cout << "Solving the " << args.n << "-Queens problem using ";

    switch( args.strategy )
    {
    case STRATEGY_SERIAL:
        cout << "a serial shared memory";
        break;
    case STRATEGY_SHARED:
        cout << "a parallel shared memory";
        break;
    case STRATEGY_DISTRIBUTED:
        cout << "a distributed memory";
        break;
    case STRATEGY_DISTRIBUTED_CUDA:
        cout << "a distributed CUDA";
        break;
    default:
        cout << "an UNKNOWN";
        break;
    }

    cout << " strategy";

    if( args.output != "" )
    {
        cout << ", outputting the results to '" << args.output << "'";
    }
    if( args.time )
    {
        cout << ", and timing the results";
    }
    cout << "." << endl;
}
