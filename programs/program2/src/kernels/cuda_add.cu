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
 * @details This kernel breaks the matrices into sub matrices of dimension @f$16
 * \times 16@f$ (the size of a block). But then what happens when the matrix is
 * not evenly divisible by the block size? For example, consider the @f$3 \times 3@f$
 * matrix below with a block size of @f$2 \times 2@f$
 * @dot
 * graph {
 *     splines=line;
 *     rankdir=LR;
 *     a -- { b d };
 *     b -- { e c };
 *     c -- f;
 *     d -- { e g };
 *     e -- { f h };
 *     g -- h;
 *     f -- i;
 *     h -- i;
 *     { rank=same; a, d, g };
 *     { rank=same; b, e, h };
 *     { rank=same; c, f, i };
 *  }
 * @enddot
 * Then we get partial blocks as shown.
 * @dot
 * graph {
 *     splines=line;
 *     //rankdir=LR;
 *     subgraph cluster1 {
 *        label="Full Block";
 *        a; b; d; e;
 *     }
 *     subgraph cluster2 {
 *        label="Partial Block";
 *        c; f;
 *     }
 *     subgraph cluster3 {
 *        label="Partial Block";
 *        g; h;
 *     }
 *     subgraph cluster4 {
 *        label="Partial Block";
 *        i;
 *     }
 *     a -- { b d };
 *     b -- { e c };
 *     c -- f;
 *     d -- { e g };
 *     e -- { f h };
 *     g -- h;
 *     f -- i;
 *     h -- i;
 *  }
 * @enddot
 * Note that I was unable to get Graphviz to draw the "Partial Block" subgraphs
 * while still keeping the nice rectangular shape of the matrix. If I were really
 * motivated, I'd draw the matrix with TikZ in LaTeX and include an SVG image, but
 * that's too much to ask for...
 *
 * There is probably a solution to prevent using an entire block to add the element
 * @f$i@f$ in the example above, but I am choosing to implement the addition simply
 * in order to get it to work, and then move on. For large matrices, this will not
 * be as much of a downside, because the matrix will (hopefully) be larger than
 * @f$16 \times 16@f$, so the impact of the additional blocks relative to the
 * cost of the whole will hopefully be less important.
 *
 * @see CudaAdditionKernelTest::MismatchedLarger and CudaAdditionKernelTest::MismatchedSmaller
 * for test cases testing matrices not evenly divisible by the block size.
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
