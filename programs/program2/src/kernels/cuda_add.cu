#include "kernels/cuda_add.h"
#include "matrix.h"
#include "validator.h"
#include <iostream>

/**
 * @brief The matrix addition CUDA kernel.
 *
 * @note Due to CUDA limitations, this cannot be a private method in the CudaAdditionKernel
 * class. Further, note that it is not possible to pass by reference to a CUDA kernel.
 * Meaning that the structure copy constructor is called every time the kernel is
 * called, and the destructor is called every time each thread finishes running
 * the kernel. This has disasterous consequences if the destructor attempts to
 * do RAII and deletes the data pointer. So the solution is to not pass structs
 * to the kernel, but instead their data pointers.
 *
 * @todo Reimplement using shared memory like the example given in
 * https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
 *
 * @param lhs The left operand.
 * @param rhs The right operand.
 * @param[out] result The matrix operation result.
 * @param rows The number of rows in each of the matrices.
 * @param cols The number of columns in each of the matrices.
 */
__global__ static void AdditionKernel( const double* lhs, const double* rhs, double* result, size_t rows, size_t cols )
{
    size_t row = blockIdx.x * blockDim.x + threadIdx.x;
    size_t col = blockIdx.y * blockDim.y + threadIdx.y;

    if( row < rows && col < cols )
    {
        result[row * cols + col] = lhs[row * cols + col] + rhs[row * cols + col];
    }
}

std::shared_ptr<Matrix_t> CudaAdditionKernel::Operation( const Matrix_t& lhs, const Matrix_t& rhs )
{
    if( !AdditionValidator( lhs, rhs ) )
    {
        std::cerr << "Dimensions (" << lhs.rows << ", " << lhs.cols << ")"
                  << " + (" << rhs.rows << ", " << rhs.cols << ")"
                  << " incompatible for addition." << std::endl;
        return std::make_shared<Matrix_t>( 0, 0 );
    }

    // Allocates the memory for the result on the host.
    std::shared_ptr<Matrix_t> result = std::make_shared<Matrix_t>( lhs.rows, lhs.cols );

    // Allocate data on the device.
    double* device_result;
    double* device_lhs;
    double* device_rhs;

    cudaMalloc( &device_result, result->elements * sizeof( double ) );
    cudaMalloc( &device_lhs, lhs.elements * sizeof( double ) );
    cudaMalloc( &device_rhs, rhs.elements * sizeof( double ) );

    // Copy the host operands to the device.
    cudaMemcpy( device_lhs, lhs.data, lhs.elements * sizeof( double ), cudaMemcpyHostToDevice );
    cudaMemcpy( device_rhs, rhs.data, rhs.elements * sizeof( double ), cudaMemcpyHostToDevice );

    dim3 threads( 16, 16 );
    //! @todo This breaks if the matrix size is not evenly divisible by the block size.
    //! @todo See https://stackoverflow.com/a/31660574/3704977
    dim3 blocks( result->rows / threads.x, result->cols / threads.y );

    // You *really* don't want to pass a struct to a CUDA kernel when the struct
    // has a copy constructor and a destructor. -____________-
    AdditionKernel<<<blocks, threads>>>( device_lhs, device_rhs, device_result, result->rows, result->cols );
    cudaDeviceSynchronize();

    // Copy the result from the device to the host.
    cudaMemcpy( result->data, device_result, result->elements * sizeof( double ), cudaMemcpyDeviceToHost );

    // Every good programmer knows every malloc() should have a corresponding free().
    cudaFree( device_result );
    cudaFree( device_lhs );
    cudaFree( device_rhs );

    return result;
}
