/**
 * @brief The main entry point and commandline argument handling portion of the program.
 *
 * @file cmain.c
 * @author Austin Gill
 * @date 2018-09-24
 */
#include "satisfiability.h"

/**
 * @brief The main entry point for the circuit satisfiability program.
 *
 * Gets the circuit to check for satisfiability from the commandline arguments
 * and brute force checks every possible input against the circuit.
 *
 * Base on compile-time `-D` defines, we will also time the satisfiability checks
 * to compare runtimes.
 *
 * @param argc The usual number of commandline arguments.
 * @param argv The usual commandline arguments.
 */
int main( int argc, char const** argv )
{
    ( void )argc;
    ( void )argv;

    return 0;
}
