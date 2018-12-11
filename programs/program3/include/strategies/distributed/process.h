#pragma once
#include "common.h"
#include <cstddef>

class Process
{
public:
    /**
     * @brief Construct a new Process object.
     *
     * @param rank This processor's rank. 0 is the master process, while all
     * others are slaves.
     * @param num_procs The total number of processors that are being utilized.
     * @param n The problem size.
     * @param verbose Whether to give diagnostic information. Defaults to false.
     */
    explicit Process( size_t rank, size_t num_procs, size_t n, bool verbose = false ) :
        rank( rank ),
        n( n ),
        verbose( verbose )
    {
        // Assign work to each of the slave processes.
        if( rank != 0 )
        {
            // Round up the chunk sizes, but the master process does no work, so
            // there are really p - 1 processes to divide the work amongst.
            size_t chunk_size = ( FACTORIALS[n] + num_procs - 1 - 1 ) / ( num_procs - 1 );
            this->begin_index = ( rank - 1 ) * chunk_size;
            // The end point is exclusive.
            this->end_index = this->begin_index + chunk_size;

            // Handle unevenly divisible problem sizes.
            if( this->begin_index > FACTORIALS[n] )
            {
                this->begin_index = FACTORIALS[n];
                this->end_index = FACTORIALS[n];
            }
            else if( this->end_index > FACTORIALS[n] )
            {
                this->end_index = FACTORIALS[n];
            }
        }

        if( this->verbose )
        {
            if( this->rank == 0 )
            {
                std::cout << "Creating Master process." << std::endl;
            }
            else
            {
                std::cout << "Creating Slave process with rank " << this->rank
                          << " handling permutations from " << this->begin_index
                          << " to " << this->end_index << std::endl;
            }
        }
    }

    virtual ~Process() = default;

    /**
     * @brief Run the process.
     *
     * @details If the processes rank is 0, it is the master process, responsible
     * for assigning work to the slaves and handling the shutdown procedure.
     *
     * The slave processes receive a work assignment and diligently, well.. slave
     * over their work until done. It was their own fault for not unionizing.
     *
     * @param screen_output Whether to print solutions to the console.
     * @param file_output An optional filename to print solutions to. The process
     * rank will be appended to the filename.
     *
     * @returns The master process returns the number of solutions, while all of
     * the slave process return the number of solutions that they found.
     */
    virtual size_t Run( bool screen_output = false, std::string file_output = "" ) = 0;

    /**
     * @brief Get the beginning index of the work this process is responsible for.
     *
     * @returns The beginning index.
     */
    size_t GetBegin()
    {
        return this->begin_index;
    }

    /**
     * @brief Get the ending index of the work this process is responsible for.
     *
     * @returns The ending index.
     */
    size_t GetEnd()
    {
        return this->end_index;
    }

    /**
     * @brief Get the problem size for this process.
     *
     * @returns The n-Queen's problem size.
     */
    size_t GetN()
    {
        return this->n;
    }

protected:
    size_t rank = 0;
    size_t n = 0;
    bool verbose = false;
    size_t begin_index = 0;
    size_t end_index = 0;
};
