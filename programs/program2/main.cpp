#include "argument_parser.h"
#include "kernels/kernel.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>

/**
 * @brief Main entry point for the application.
 *
 * @returns 0 to indicate success.
 */
int main( int argc, const char** argv )
{
    ArgumentParser parser( argc, argv );
    ArgumentParser::Args_t args = parser.ParseArgs();

    Kernel kernel( args.operation, args.kernel );

    Matrix_t lhs( args.left_input );
    Matrix_t rhs( args.right_input );

    Matrix_t res = kernel.Operation( lhs, rhs );

    if( args.output )
    {
        res.Serialize( args.output_file );
    }
    else
    {
        res.Print();
    }

    return 0;
}
