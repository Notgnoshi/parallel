#include "args.h"
#include "kernels/kernel.h"
#include "kernels/kernel_factory.h"
#include "matrix.h"
#include <iostream>

/**
 * @brief Main entry point for the application.
 *
 * @returns 0 to indicate success.
 */
int main( int argc, const char** argv )
{
    ArgumentParser parser( argc, argv );
    ArgumentParser::Args_t args = parser.ParseArgs();

    KernelFactory factory( args.operation, args.kernel );
    auto kernel = factory.GetKernel();

    Matrix_t lhs( args.left_input );
    Matrix_t rhs( args.right_input );

    std::shared_ptr<Matrix_t> res = kernel->Operation( lhs, rhs );

    if( *res == Matrix_t( 0, 0 ) )
    {
        std::cerr << "Operation received invalid dimensions." << std::endl;
    }

    if( args.output )
    {
        res->Serialize( args.output_file );
    }
    else
    {
        res->Print();
    }

    return 0;
}
