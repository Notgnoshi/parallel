#include "incr_kernel.cuh"
#include "incr_wrapper.h"

__global__ void incr_kernel( double* a, const double* b, size_t n )
{
    for( size_t i = 0; i < n; ++i )
    {
        a[i] = a[i] + b[i];
    }
}

void incr( double* a, const double* b, size_t n )
{
    double* _a;
    double* _b;

    cudaMalloc( &_a, n * sizeof( double ) );
    cudaMalloc( &_b, n * sizeof( double ) );

    cudaMemcpy( _a, a, n * sizeof( double ), cudaMemcpyHostToDevice );
    cudaMemcpy( _b, b, n * sizeof( double ), cudaMemcpyHostToDevice );

    //! @note This is essentially serial, with *loads* of memcpy overhead.
    incr_kernel<<<1, 1>>>( _a, _b, n );

    cudaMemcpy( a, _a, n * sizeof( double ), cudaMemcpyDeviceToHost );

    cudaFree( _a );
    cudaFree( _b );
}
