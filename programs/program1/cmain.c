/**
 * @file cmain.c
 * @author Austin Gill (atgill@protonmail.com)
 * @brief The main entry point and commandline argument handling portion of the program.
 *
 * @todo Produce PDF documentation.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "satisfiability.h"

/**
 * @brief Prints the program usage statement.
 *
 * @param prog_name The name of the program, taken from the commandline arguments.
 */
void usage( char const* prog_name )
{
    printf( "Computes circuit satisfiability for digital circuits.\n" );
    printf( "Note: only circuits 1 and 2 are supported.\n\n" );
    printf( "Usage: %s <circuit>\n", prog_name );
}

/**
 * @brief Get the circuit from the commandline arguments.
 *
 * @note This function will exit the program and print a usage statement on failure.
 *
 * @returns The circuit number from the commandline arguments, if it's valid.
 */
int32_t get_circuit( int32_t argc, char const** argv )
{
    int32_t circuit = 0;
    if( argc == 2 )
    {
        circuit = atoi( argv[1] );
    }

    if( circuit == 1 || circuit == 2 )
    {
        return circuit;
    }

    usage( argv[0] );
    exit( 1 );
}

/**
 * @brief The main entry point for the circuit satisfiability program.
 *
 * Gets the circuit to check for satisfiability from the commandline arguments
 * and brute force checks every possible input against the circuit.
 *
 * Based on compile-time `-D` defines, we will time the satisfiability checks
 * to compare runtimes.
 *
 * Based on compile-time `-D` defines, we will also compare static and dynamic
 * scheduling methods.
 *
 * @param argc The usual number of commandline arguments.
 * @param argv The usual commandline arguments.
 */
int main( int argc, char const** argv )
{
    int32_t circuit = get_circuit( argc, argv );

    // A more elegant solution would be to store the function pointers in a map.
    if( circuit == 1 )
    {
        check_circuit( circuit_one );
    }
    else if( circuit == 2 )
    {
        check_circuit( circuit_two );
    }

    return 0;
}
