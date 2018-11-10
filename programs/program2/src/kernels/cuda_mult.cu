#include "kernels/cuda_mult.h"
#include "matrix.h"
#include "validator.h"
#include <iostream>

//! @brief Use @f$16 \times 16@f$ blocks of threads.
//! @details Use a `#define` so that it's accessable in both device and host code.
#define BLOCK_XDIM 16
#define BLOCK_YDIM 16
#define BLOCK_ZDIM 1
//! @todo Document.
static const dim3 BLOCK_SIZE( BLOCK_XDIM, BLOCK_YDIM, BLOCK_ZDIM );

/**
 * @brief A trivially easy-to-copy struct to hold the matrices on the device.
 *
 * @details The Matrix_t struct defines a copy-constructor, destructor, etc, so
 * it doesn't work well for passing to CUDA kernel calls. So use this struct to
 * hold trivially-easy-to-copy device fields so that the kernel calls don't get
 * so nasty.
 */
struct DeviceMatrix_t
{
    //! @brief The width of the matrix.
    size_t cols;
    //! @brief The height of the matrix.
    size_t rows;
    //! @brief The stride in the 1D array that the 2D data is stored in.
    size_t stride;
    //! @brief The matrix data stored in row-major order.
    double* data;
};

/**
 * @brief Given a matrix, get the element at (row, col).
 *
 * @param matrix The DeviceMatrix_t holding the data to access.
 * @param row ...
 * @param col ...
 * @returns the element at the given location.
 */
__device__ static double GetElement( const DeviceMatrix_t matrix, size_t row, size_t col )
{
    return matrix.data[row * matrix.stride + col];
}

/**
 * @brief Given a matrix, set the value at (row, col).
 *
 * @param[out] matrix The DeviceMatrix_t matrix to modify.
 * @param row ...
 * @param col ...
 * @param value The value to insert at (row, col)
 */
__device__ static void SetElement( DeviceMatrix_t matrix, size_t row, size_t col, double value )
{
    matrix.data[row * matrix.stride + col] = value;
}

/**
 * @brief Get the @f$(row, col)@f$th sub matrix in the given matrix.
 *
 * @param matrix The matrix to extract sub matrices from.
 * @param row Which row of sub-matrices to extract the sub matrix from.
 * @param col Which column of sub-matrices to extract the sub matrix from.
 * @returns A submatrix from the given matrix.
 */
__device__ static DeviceMatrix_t GetSubMatrix( const DeviceMatrix_t matrix, size_t row, size_t col )
{
    DeviceMatrix_t sub;
    //! @todo for non-square blocks, these will flip depending on lhs or rhs.
    sub.cols = BLOCK_XDIM;
    sub.rows = BLOCK_YDIM;
    sub.stride = matrix.stride;
    //! @todo Figure out if it's (x, y) or (y, x) for non-square blocks.
    sub.data = &matrix.data[matrix.stride * BLOCK_YDIM * row + BLOCK_XDIM * col];

    return sub;
}

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
 * @details Using the `MatMulKernel` in the CUDA documentation for inspiration,
 * this kernel uses shared memory and submatrices to implement matrix multiplication.
 * The biggest difference between this implementation and that in the CUDA
 * documentation is that this will work for matrices that are not evenly divisible
 * by the block size.
 *
 * @see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#shared-memory__matrix-multiplication-shared-memory for a good picture of what I'm doing.
 *
 * @todo Document.
 */
__global__ static void MultiplicationKernel( const DeviceMatrix_t lhs, const DeviceMatrix_t rhs, DeviceMatrix_t result )
{
    // Each block of threads computes a submatrix of the result.
    DeviceMatrix_t sub = GetSubMatrix( result, blockIdx.y, blockIdx.x );

    // Each thread computes one value in the submatrix.
    double value = 0;

    // Loop over the submatrices of lhs and rhs to compute sub.
    for( size_t i = 0; i < ( BLOCK_XDIM + lhs.cols - 1 ) / BLOCK_XDIM; ++i )
    {
        // Fixed row and variable columm.
        const DeviceMatrix_t lhs_sub = GetSubMatrix( lhs, blockIdx.y, i );
        // Fixed column and variable row.
        const DeviceMatrix_t rhs_sub = GetSubMatrix( rhs, i, blockIdx.x );

        // Share lhs_sub and rhs_sub across the block.
        __shared__ double left_block[BLOCK_YDIM][BLOCK_XDIM];
        __shared__ double right_block[BLOCK_YDIM][BLOCK_XDIM];

        //! @note Each thread loads an element of the result submatrix that their
        //! block is responsible for computing. However, note that since the blocks
        //! can cover more than the matrix memory, this will access memory we do
        //! not own. This is bad, but I'd rather do that than add more branches.
        //! We don't modify the data, so it's not *that* bad... (it actually is).
        left_block[threadIdx.y][threadIdx.x] = GetElement( lhs_sub, threadIdx.y, threadIdx.x );
        right_block[threadIdx.y][threadIdx.x] = GetElement( rhs_sub, threadIdx.y, threadIdx.x );

        // Finish loading the shared memory before proceeding.
        __syncthreads();

        //! @note You have no idea how long it took, or how many pages of graph paper
        //! I went through to get this formula.
        const size_t lim = BLOCK_YDIM - ( ( ( i + 1 ) * BLOCK_YDIM ) % lhs.cols ) % BLOCK_YDIM;
        for( size_t j = 0; j < lim; ++j )
        {
            value += left_block[threadIdx.y][j] * right_block[j][threadIdx.x];
        }

        // Make sure computation is finished before loading new submatrices into shared memory.
        __syncthreads();
    }

    // Convert coordinates of the submatrix and the location inside the submatrix
    // into the coordinates of the whole result matrix.
    size_t row = blockIdx.y * blockDim.y + threadIdx.y;
    size_t col = blockIdx.x * blockDim.x + threadIdx.x;

    // Only set the submatrix value if it lies inside the result.
    if( row < result.rows && col < result.cols )
    {
        // Each thread writes one element in their block's submatrix.
        SetElement( sub, threadIdx.y, threadIdx.x, value );
    }
}

std::shared_ptr<Matrix_t> CudaMultiplicationKernel::Operation( const Matrix_t& lhs, const Matrix_t& rhs )
{
    if( !MultiplicationValidator( lhs, rhs ) )
    {
        // std::cerr << "Dimensions (" << lhs.rows << ", " << lhs.cols << ")"
        //           << " * (" << rhs.rows << ", " << rhs.cols << ")"
        //           << " incompatible for multiplication." << std::endl;
        return std::make_shared<Matrix_t>( 0, 0 );
    }

    // Allocate memory for the result on the host.
    auto result = std::make_shared<Matrix_t>( lhs.rows, rhs.cols );

    DeviceMatrix_t _lhs;
    DeviceMatrix_t _rhs;
    DeviceMatrix_t _result;

    _lhs.cols = lhs.cols;
    _lhs.stride = lhs.cols;
    _lhs.rows = lhs.rows;
    _rhs.cols = rhs.cols;
    _rhs.stride = rhs.cols;
    _rhs.rows = rhs.rows;
    _result.cols = result->cols;
    _result.stride = result->cols;
    _result.rows = result->rows;

    cudaMalloc( &_result.data, result->elements * sizeof( double ) );
    cudaMalloc( &_lhs.data, lhs.elements * sizeof( double ) );
    cudaMalloc( &_rhs.data, rhs.elements * sizeof( double ) );

    // Copy the operands to the device.
    cudaMemcpy( _lhs.data, lhs.data, lhs.elements * sizeof( double ), cudaMemcpyHostToDevice );
    cudaMemcpy( _rhs.data, rhs.data, rhs.elements * sizeof( double ), cudaMemcpyHostToDevice );

    dim3 grid_size(
        ( BLOCK_SIZE.x + result->cols - 1 ) / BLOCK_SIZE.x,
        ( BLOCK_SIZE.y + result->rows - 1 ) / BLOCK_SIZE.y,
        1 );
    MultiplicationKernel<<<grid_size, BLOCK_SIZE>>>( _lhs, _rhs, _result );

    cudaDeviceSynchronize();

    // Copy the result from the device to the host.
    cudaMemcpy( result->data, _result.data, result->elements * sizeof( double ), cudaMemcpyDeviceToHost );

    // Every good programmer knows every malloc() should have a corresponding free().
    cudaFree( _result.data );
    cudaFree( _lhs.data );
    cudaFree( _rhs.data );

    return result;
}
