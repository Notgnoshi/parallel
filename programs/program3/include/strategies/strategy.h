#pragma once

#include "args.h"
#include <cstddef>
#include <mpi.h>
#include <string>

/**
 * @brief An abstract base class to represent a solution strategy to the @f$n@f$ queens problem.
 */
class Strategy
{
public:
    /**
     * @brief Default-destroy the Strategy object.
     */
    virtual ~Strategy() = default;

    /**
     * @brief Initialize this MPI strategy.
     *
     * @param argc A pointer to the usual argc.
     * @param argv A pointer to the usual argv.
     */
    virtual void Initialize( int* argc, const char*** argv )
    {
        MPI_Init( argc, const_cast<char***>( argv ) );
        MPI_Comm_size( MPI_COMM_WORLD, &this->num_procs );
        MPI_Comm_rank( MPI_COMM_WORLD, &this->rank );
        MPI_Get_processor_name( this->proc_name, &this->proc_name_length );
    }

    /**
     * @brief Clean up after running this MPI strategy.
     */
    virtual void Finalize()
    {
        MPI_Finalize();
    }

    /**
     * @brief Get the Rank of this process.
     *
     * @returns The rank of the current process.
     */
    int GetRank()
    {
        return this->rank;
    }

    /**
     * @brief Run the Strategy on a problem of the given size.
     *
     * @param n      The chessboard size.
     * @param output Whether to output solutions to the screen. Defaults to true.
     */
    virtual void Run( ArgumentParser::Args_t args ) = 0;

protected:
    int num_procs = 0;
    int rank = 0;
    int proc_name_length = 0;
    char proc_name[MPI_MAX_PROCESSOR_NAME] = {0};
};
