#pragma once

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
     * @brief Construct a new Strategy object.
     *
     * @details If the `time` parameter is `true`, the Strategy will time the operation
     * and print the result to stderr in the format `__PRETTY_FUNCTION__: <time> ms`.
     *
     * @param file_output If nonempty, the filename to save the outputs to.
     * @param screen_output Whether to output solutions to the screen. Defaults to false.
     * @param time Whether or not to time the solution. Defaults to false.
     * @param verbose Whether or not to output diagnostic information.
     */
    explicit Strategy( std::string file_output = "", bool screen_output = false, bool time = false, bool verbose = false ) :
        file_output( file_output ),
        screen_output( screen_output ),
        time( time ),
        verbose( verbose )
    {}

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

    size_t GetProcs()
    {
        return static_cast<size_t>( this->num_procs );
    }

    bool GetVerbose()
    {
        return this->verbose;
    }

    /**
     * @brief Run the Strategy on a problem of the given size.
     *
     * @param n      The chessboard size.
     *
     * @returns The number of solutions found.
     */
    virtual size_t Run( size_t n ) = 0;

protected:
    std::string file_output = "";
    bool screen_output = false;
    bool time = false;
    bool verbose = false;
    int num_procs = 0;
    int rank = 0;
    int proc_name_length = 0;
    char proc_name[MPI_MAX_PROCESSOR_NAME] = {0};
};
