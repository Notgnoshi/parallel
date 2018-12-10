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
    explicit Process( int rank, size_t num_procs, size_t n, bool verbose = false ) :
        rank( rank ),
        n( n ),
        verbose( verbose )
    {
        //! @todo What if the problem size is not evenly divisible by the number
        //! of processes?
        //! There are two possibilities:
        //! 1. There are not enough processes.
        //! 2. There are too many processes.
        //!
        //! In the first case, give the extra work to the end process(es?) because
        //! the majority of the work is at the beginning of the permutations.
        //! In the second case, some processes need to idle.

        // The master process does no work, so there are really p - 1 processes
        // to divide the work amongst.
        this->begin_index = FACTORIALS[n] * ( rank - 1 ) / ( num_procs - 1 );
        // The end point is exclusive.
        this->end_index = this->begin_index + FACTORIALS[n] / ( num_procs - 1 );

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
     * @returns The master process returns the number of solutions, while all of
     * the slave process return the number of solutions that they found.
     */
    virtual size_t Run() = 0;

protected:
    int rank = 0;
    size_t n = 0;
    bool verbose = false;
    size_t begin_index = 0;
    size_t end_index = 0;
};
