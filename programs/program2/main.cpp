#include "argument_parser.h"
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

    (void)args;

    return 0;
}
