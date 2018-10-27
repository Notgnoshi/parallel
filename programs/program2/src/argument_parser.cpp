#include "argument_parser.h"
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>

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
    std::vector<std::string> positionals;

    // Find all of the optional arguments, skipping the first (executable name).
    for( auto arg = this->argv.begin() + 1; arg != this->argv.end(); ++arg )
    {
        if( *arg == "-h" || *arg == "--help" )
        {
            this->Usage();
            exit( 0 );
        }
        else if( *arg == "-o" || *arg == "--output" )
        {
            // Consume both the flag and its argument.
            ++arg;
            args.output = true;
            args.output_file = *arg;
        }
        else if( *arg == "-k" || *arg == "--kernel" )
        {
            ++arg;
            try
            {
                args.kernel = static_cast<Kernel_e>( std::stoi( *arg ) );
            } catch( ... )
            {
                std::cerr << "Failed to parse kernel." << std::endl;
                this->Usage();
                exit( 1 );
            }
        }
        else
        {
            // Save the positional arguments in the order they are encountered.
            positionals.push_back( *arg );
        }
    }

    if( positionals.size() != 3 )
    {
        std::cerr << "Missing positional argument." << std::endl;
        this->Usage();
        exit( 1 );
    }

    if( positionals[0] == "MVM" )
    {
        args.operation = OPERATION_VECTOR_MULTIPLICATION;
    }
    else if( positionals[0] == "MMA" )
    {
        args.operation = OPERATION_ADDITION;
    }
    else
    {
        std::cerr << "Failed to parse operation." << std::endl;
        this->Usage();
        exit( 1 );
    }

    if( FileExists( positionals[1] ) )
    {
        args.left_input = positionals[1];
    }
    else
    {
        std::cerr << "File " << positionals[1] << " does not exist." << std::endl;
        exit( 1 );
    }

    if( FileExists( positionals[2] ) )
    {
        args.right_input = positionals[2];
    }
    else
    {
        std::cerr << "File " << positionals[2] << " does not exist." << std::endl;
        exit( 1 );
    }

    return args;
}

void ArgumentParser::Usage()
{
    std::cout << "Usage: " << this->argv[0] << " ";
    std::cout << "[--help] [--output] [--kernel]"
              << " ";
    std::cout << "<operation> <input1> <input2>";
    std::cout << std::endl
              << std::endl;
    std::cout << "CUDA accelerated matrix operations.";
    std::cout << std::endl
              << std::endl;

    std::cout << "positional arguments:" << std::endl;
    std::cout << std::left << std::setw( 15 ) << "  operation"
              << "One of {MVM, MMA}" << std::endl;
    std::cout << std::left << std::setw( 15 ) << "  input1"
              << "An input file for the left operand" << std::endl;
    std::cout << std::left << std::setw( 15 ) << "  input2"
              << "An input file for the right operand" << std::endl;
    std::cout << std::endl
              << std::endl;

    std::cout << "optional arguments:" << std::endl;
    std::cout << std::left << std::setw( 15 ) << " -h, --help"
              << "Show this help message and exit" << std::endl;
    std::cout << std::left << std::setw( 15 ) << " -o, --output"
              << "Save the output to the given file" << std::endl;
    std::cout << std::left << std::setw( 15 ) << " -k, --kernel"
              << "The kernel to use for the given operation. Must be one of" << std::endl;
    std::cout << std::left << std::setw( 15 ) << " "
              << "    0 - Use the default kernel for the given operation." << std::endl;
    std::cout << std::left << std::setw( 15 ) << " "
              << "    1 - Do not use a CUDA kernel. Perform the operation on the CPU." << std::endl;
}

bool ArgumentParser::FileExists( const std::string& filename )
{
    std::ifstream file( filename );
    return file.good();
}
