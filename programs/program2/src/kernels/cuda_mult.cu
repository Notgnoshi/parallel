#include "kernels/cuda_mult.h"
#include "matrix.h"
#include "validator.h"
#include <iostream>

/**
 * @brief The multiplication kernel.
 *
 * @note Due to CUDA limitations, this cannot be a private method in the CudaAdditionKernel
 * class. Further, note that it is not possible to pass by reference to a CUDA kernel.
 * Meaning that the structure copy constructor is called every time the kernel is
 * called, and the destructor is called every time each thread finishes running
 * the kernel. This has disasterous consequences if the destructor attempts to
 * do RAII and deletes the data pointer. So the solution is to not pass structs
 * to the kernel, but instead their data pointers.
 *
 * @see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory
 * for a Matrix-Matrix multiplication kernel that takes advantage of shared memory.
 *
 * @param lhs The left operand.
 * @param rhs The right operand.
 * @param[out] result The matrix operation result.
 * @param left_rows The number of rows in the left matrix.
 * @param left_cols The number of columns in the left matrix.
 * @param right_rows The number of rows in the right vector.
 * @param right_cols The number of columns in the right vector.
 */
__global__ static void MultiplicationKernel( const double* lhs, const double* rhs, double* result,
                                             size_t left_rows, size_t left_cols,
                                             size_t right_rows, size_t right_cols )
{
    //! @todo Implement.
}

std::shared_ptr<Matrix_t> CudaMultiplicationKernel::Operation( const Matrix_t& lhs, const Matrix_t& rhs )
{
    if( !MultiplicationValidator( lhs, rhs ) )
    {
        std::cerr << "Dimensions (" << lhs.rows << ", " << lhs.cols << ")"
                  << " * (" << rhs.rows << ", " << rhs.cols << ")"
                  << " incompatible for multiplication." << std::endl;
        return std::make_shared<Matrix_t>( 0, 0 );
    }

    // Allocate memory for the result on the host.
    auto result = std::make_shared<Matrix_t>( lhs.rows, rhs.cols );

    // Allocate memory for the input and output on the device.
    double* device_result;
    double* device_lhs;
    double* device_rhs;

    cudaMalloc( &device_result, result->elements * sizeof( double ) );
    cudaMalloc( &device_lhs, lhs.elements * sizeof( double ) );
    cudaMalloc( &device_rhs, rhs.elements * sizeof( double ) );

    // Copy the operands to the device.
    cudaMemcpy( device_lhs, lhs.data, lhs.elements * sizeof( double ), cudaMemcpyHostToDevice );
    cudaMemcpy( device_rhs, rhs.data, rhs.elements * sizeof( double ), cudaMemcpyHostToDevice );

    //! @todo Determine the block and grid sizes.
    MultiplicationKernel<<<1, 1>>>( device_lhs, device_rhs, device_result, lhs.rows, lhs.cols, rhs.rows, rhs.cols );

    cudaDeviceSynchronize();

    // Copy the result from the device to the host.
    cudaMemcpy( result->data, device_result, result->elements * sizeof( double ), cudaMemcpyDeviceToHost );

    // Every good programmer knows every malloc() should have a corresponding free().
    cudaFree( device_result );
    cudaFree( device_lhs );
    cudaFree( device_rhs );

    return result;
}
