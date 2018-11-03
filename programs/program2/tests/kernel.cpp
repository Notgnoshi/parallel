#include "kernel.h"
#include "argument_parser.h"
#include "kernels/kernel.h"
#include "matrix.h"

void KernelTest::DefaultMultOperation()
{
    Kernel kernel( ArgumentParser::OPERATION_VECTOR_MULTIPLICATION, ArgumentParser::KERNEL_DEFAULT );

    Matrix_t lhs( "./matrices/4x4_ones.mat" );
    Matrix_t rhs( "./matrices/4x1_ones.mat" );

    const Matrix_t expected( "./matrices/4x1_fours.mat" );

    Matrix_t result = kernel.Operation( lhs, rhs );

    CPPUNIT_ASSERT( result == expected );
}

void KernelTest::DefaultAddOperation()
{
    Kernel kernel( ArgumentParser::OPERATION_ADDITION, ArgumentParser::KERNEL_DEFAULT );

    Matrix_t lhs( "./matrices/3x4_ones.mat" );
    Matrix_t rhs( "./matrices/3x4_twos.mat" );

    const Matrix_t expected( "./matrices/3x4_threes.mat" );

    Matrix_t result = kernel.Operation( lhs, rhs );

    CPPUNIT_ASSERT( result == expected );
}
