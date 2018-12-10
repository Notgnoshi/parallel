#pragma once
#include "strategies/distributed/process.h"
#include <memory>

std::unique_ptr<Process> GetProcess( int rank, size_t num_procs, size_t n, bool verbose = false );
