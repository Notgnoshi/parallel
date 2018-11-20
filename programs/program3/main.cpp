#include <cstdio>
#include <cstdlib>
#include <mpi.h>

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
int main( int argc, char** argv )
{
    int nprocs, rank, namelen;
    char processor_name[MPI_MAX_PROCESSOR_NAME];

    MPI_Init( &argc, &argv );
    MPI_Comm_size( MPI_COMM_WORLD, &nprocs );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    MPI_Get_processor_name( processor_name, &namelen );

    printf( "Process %d out of %d on %s\n", rank, nprocs, processor_name );

    MPI_Finalize();

    return 0;
}
